module {
  func.func @main(%arg0: tensor<128x128xsi8>, %arg1: tensor<128x128xsi8>) -> tensor<128x128xf16> {
    %scale = "mps.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %zero = "mps.constant"() <{value = dense<0> : tensor<si8>}> : () -> tensor<si8>
    %min = "mps.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %0 = "mpsx.quantized_matmul"(%arg0, %scale, %zero, %min, %arg1, %scale, %zero, %min) {
      input_quant_params_axis = 0 : si32,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
      output_type = f16,
      transpose_lhs = false,
      transpose_rhs = false,
      weights_quant_params_axis = 0 : si32
    } : (tensor<128x128xsi8>, tensor<f32>, tensor<si8>, tensor<f32>, tensor<128x128xsi8>, tensor<f32>, tensor<si8>, tensor<f32>) -> tensor<128x128xf16>
    return %0 : tensor<128x128xf16>
  }
}
