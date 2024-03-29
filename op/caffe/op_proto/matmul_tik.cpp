#include "matmul_tik.h"
#include <string>
#include <vector>

namespace ge {

IMPLEMT_VERIFIER(MatmulTik, MatmulTikVerify)
{
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT32);
  support_list.push_back(DT_INT8);
  support_list.push_back(DT_UINT8);

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatmulTikInferShape)
{
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    ge::TensorDesc inputTensorDescX = op.GetInputDesc("x1");
    ge::TensorDesc inputTensorDescY = op.GetInputDesc("x2");

    ge::Shape shapeX = inputTensorDescX.GetShape();
    ge::Shape shapeY = inputTensorDescY.GetShape();

    DataType dtype = inputTensorDescX.GetDataType();

    bool transposeA = false;


    bool transposeB = false;

    
    std::vector<int64_t> dimVector;
    dimVector.push_back(shapeX.GetDim(0));
    dimVector.push_back(shapeY.GetDim(1));


    ge::Shape outputShape(dimVector);

    tensordesc_output.SetShape(outputShape);
    tensordesc_output.SetDataType(op.GetInputDesc("x1").GetDataType());
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(MatmulTik, MatmulTikInferShape);

//Registered verify function
VERIFY_FUNC_REG(MatmulTik, MatmulTikVerify);
}