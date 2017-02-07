#include "parrots/dnn/layerbase.hpp"
#include "parrots/datablas.hpp"

#include "ctc.h"


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

      PARROTS_CHECKARGS(ins[0].elemType().getPrim() == PARROTS_FLOAT32);
      PARROTS_CHECKARGS(ins[1].elemType().getPrim() == PARROTS_INT32);

      //TODO: check dimension here

      InferredDataSpecs spec;
      spec.topSpecs.push_back(DataSpec::scalar(ins[0].elemType()));

      return spec;
    }

    layer_uptr_t createLayer(LayerContext& ctx, DeviceProxy& deviceProxy, JobType jobType,
                             const DataSpecList& inputs, const DataSpecList& outputs) const override;
};

class CTCLossLayer : public Layer<CTCLossLayerProto> {


public:
    CTCLossLayer(const CTCLossLayerProto& proto)
    : Layer<CTCLossLayerProto>(proto){

    }

    void forward(const LayerContext& ctx,
                 const std::vector<LayerInput>& bottoms,
                 const std::vector<LayerOutput>& tops) override {
      auto grad_buffer = ctx.local.get("grad_buffer", bottoms[0].dc.spec());
      float loss_weight = getScalar<float>(tops[0].dc.gradObject());
    }

    void backward(const LayerContext& ctx,
                 const std::vector<LayerInput>& bottoms,
                 const std::vector<LayerInput>& tops,
                 const std::vector<LayerGradOutput>& bottomGrads) override {
      auto grad_buffer = ctx.local.get("grad_buffer", bottoms[0].dc.spec());

      float alpha = bottomGrads[0].alpha;
      float beta = bottomGrads[0].beta;
      float loss_weight = getScalar<float>(tops[0].dc.gradObject());

      if (beta != 0) {
        blas::scale(beta, *bottomGrads[0].pObj);
      }else{
        bottomGrads[0].pObj->setZeros();
      }
      auto scale = loss_weight * alpha;
      blas::axpy(scale, grad_buffer, *bottomGrads[0].pObj);
    }
};

layer_uptr_t CTCLossLayerProto::createLayer(LayerContext &ctx, DeviceProxy &deviceProxy, JobType jobType,
                                            const DataSpecList &inputs, const DataSpecList &outputs) const {
    ctx.local.reserve("grad_buffer", inputs[0]);
    return layer_uptr_t(new CTCLossLayer(*this));
}

PARROTS_AUTO_REGISTER_LAYER("CTCLoss", CTCLossLayerProto);

}
}