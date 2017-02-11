#include "parrots/dnn/layerbase.hpp"
#include "parrots/datablas.hpp"
#include "parrots/datareduce.hpp"

#include "ctc.h"
#include "helper.hpp"
#include <thread>

namespace parrots {

namespace dnn {


class CTCLossLayer;

class CTCLossLayerProto : public LayerProto{
public:
    CTCLossLayerProto(const SSElement& attrs) : LayerProto(attrs){
      PARROTS_CHECKARGS(attrs.empty());
    }

    PARROTS_LAYER_ISLOSS;
    PARROTS_SUPPORT_PARTIAL_BACKWARD;

    InferredDataSpecs inferDataSpecs(const std::vector<DataSpec>& ins, size_t nparams) const override {
      PARROTS_CHECKARGS(nparams == 0);
      PARROTS_CHECKARGS(ins.size() == 2);

      PARROTS_CHECKARGS(ins[0].elemType() == float32_);
      PARROTS_CHECKARGS(ins[1].elemType() == int32_);

      //check dimensions
      PARROTS_CHECKARGS(ins[0].ndims() == 3);
      PARROTS_CHECKARGS(ins[1].ndims() == 2);

      //batch sizes must be equal
      PARROTS_CHECKARGS(ins[0].numSlices() == ins[1].numSlices());

      // num_label must be less than time steps
      PARROTS_CHECKARGS(ins[0].dim(1) >= ins[1].dim(0));

      InferredDataSpecs spec;
      spec.topSpecs.push_back(DataSpec::scalar(ins[0].elemType()));

      return spec;
    }

    layer_uptr_t createLayer(LayerContext& ctx, DeviceProxy& deviceProxy, JobType jobType,
                             const DataSpecList& inputs, const DataSpecList& outputs) const override;
};

class CTCLossLayer : public Layer<CTCLossLayerProto> {

    ctcOptions ctc_opts_;


    int num_class_;
    int batchsize_;

    size_t workspace_size_;
    DeviceProxy& device_p_;

    // host label
    DataObject hostLabelObj_;

    // buffer for storing dimensions on the device
    DataObject dimensions_;

    DataObject label_lengths_;
    DataObject input_lengths_;

public:
    CTCLossLayer(const CTCLossLayerProto& proto, DeviceProxy& deviceProxy,
                 const DataSpecList& inputs, const DataSpecList& outputs)
    : Layer<CTCLossLayerProto>(proto),
      device_p_(deviceProxy),
      hostLabelObj_(getHostProxy(), inputs[1]),
      dimensions_(device_p_, DataSpec::array(uint64_, inputs[0].ndims())){

      // setup ctc constructs
      ctc_opts_.blank_label = 0;
      ctc_opts_.loc = (deviceProxy.arch() == Arch::X86)?CTC_CPU:CTC_GPU;
      if (ctc_opts_.loc == CTC_GPU){
        ctc_opts_.stream = cudaStreamLegacy;
      }else{
        ctc_opts_.num_threads = std::thread::hardware_concurrency();
      }

      // get the dimensions
      num_class_ = inputs[0].dim(0);
      batchsize_ = inputs[0].dim(2);

      label_lengths_ = DataObject(getHostProxy(), DataSpec::array(type_<int>(), (size_t)batchsize_));
      input_lengths_ = DataObject(getHostProxy(), DataSpec::array(type_<int>(), (size_t)batchsize_));

      for (int i = 0; i < batchsize_; ++i) {
        label_lengths_.tdata<int>()[i] = inputs[1].dim(0);
        input_lengths_.tdata<int>()[i] = inputs[0].dim(1);
      }

      ctcStatus_t status = get_workspace_size(label_lengths_.tdata<int>(), input_lengths_.tdata<int>(), num_class_, batchsize_,
                                              ctc_opts_, &workspace_size_);

      // reserve memory for quick buffer
      device_p_.reserveWorkSpace(workspace_size_);

      auto dims = box(inputs[0].dims(), arrshape(inputs[0].ndims()));
      copy(dimensions_, dims);

    }

    void forward(const LayerContext& ctx,
                 const std::vector<LayerInput>& bottoms,
                 const std::vector<LayerOutput>& tops) override {
      auto grad_buffer = ctx.local.get("grad_buffer", bottoms[0].dc.spec());
      auto act_buffer = ctx.local.get("act_buffer", bottoms[0].dc.spec());

      float loss_weight = getScalar<float>(tops[0].dc.gradObject());

      DataObject costs(getHostProxy(), DataSpec::array(type_<float>(), batchsize_));

      ::parrots::internal::QuickBuffer workspace(device_p_, workspace_size_);

      std::cout<<"buffer setup\n";

      // prepare data
      copy(hostLabelObj_, bottoms[1].dc.valueObject());

      std::cout<<"data copied\n";

      // warp ctc uses a interleaved data layout, we have to first permute the activation tensor to fit its layout
      permute_dimension(dimensions_.tdata<size_t>(), act_buffer.ndims(), act_buffer.size(),
                        act_buffer.tdata<float>(), 0, (float)0,
                        bottoms[0].dc.valueObject().tdata<float>(), 1, (float)0,
                        ctc_opts_
      );
      std::cout<<"data permuted\n";

      auto status = compute_ctc_loss(act_buffer.tdata<float>(), grad_buffer.tdata<float>(),
                                     hostLabelObj_.tdata<int>(),
                                     label_lengths_.tdata<int>(), input_lengths_.tdata<int>(), num_class_, batchsize_,
                                     costs.tdata<float>(), workspace.data(), ctc_opts_);

      if (status != 0){
        std::cout<<ctcGetStatusString(status)<<"\n";
      }

      setScalar(tops[0].dc.valueObject(), mean(costs));
      std::cout<<"loss: "<<mean(costs)<<"\n";

      std::cout<<"forward done\n";
    }

    void backward(const LayerContext& ctx,
                 const std::vector<LayerInput>& bottoms,
                 const std::vector<LayerInput>& tops,
                 const std::vector<LayerGradOutput>& bottomGrads) override {

      std::cout<<"backward started\n";
      auto grad_buffer = ctx.local.get("grad_buffer", bottoms[0].dc.spec());

      float alpha = bottomGrads[0].alpha;
      float beta = bottomGrads[0].beta;
      float loss_weight = getScalar<float>(tops[0].dc.gradObject());


      std::cout<<"backward permuting\n";
      auto scale = loss_weight * alpha;
      permute_dimension(dimensions_.tdata<size_t>(), grad_buffer.ndims(), grad_buffer.size(),
                        bottomGrads[0].pObj->tdata<float>(), 0, beta,
                        grad_buffer.tdata<float>(), 1, scale,
                        ctc_opts_);
    }
};

layer_uptr_t CTCLossLayerProto::createLayer(LayerContext &ctx, DeviceProxy &deviceProxy, JobType jobType,
                                            const DataSpecList &inputs, const DataSpecList &outputs) const {
    ctx.local.reserve("grad_buffer", inputs[0]);
    ctx.local.reserve("act_buffer", inputs[0]);
    return layer_uptr_t(new CTCLossLayer(*this, deviceProxy, inputs, outputs));
}

PARROTS_AUTO_REGISTER_LAYER("CTCLoss", CTCLossLayerProto);

}
}