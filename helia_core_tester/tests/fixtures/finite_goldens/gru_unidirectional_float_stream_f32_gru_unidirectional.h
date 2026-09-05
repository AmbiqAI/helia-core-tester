#ifndef GRU_UNIDIRECTIONAL_FLOAT_STREAM_F32_GRU_UNIDIRECTIONAL_H
#define GRU_UNIDIRECTIONAL_FLOAT_STREAM_F32_GRU_UNIDIRECTIONAL_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"

static const float32_t gru_unidirectional_float_stream_f32_input_tensor[] = {
    -0.552010536f, 0.700960755f, 0.046279561f, -0.580457091f, -0.335887522f, 0.995170832f, -0.769025385f, 0.636148214f, 0.134862736f, -0.883449435f, -0.229363516f, -0.605748594f
};

static const float32_t gru_unidirectional_float_stream_f32_expected_output[] = {
    -0.115517452f, 0.095142819f, 0.074682564f, -0.202843562f, 0.013375841f, 0.161503345f, 0.01629303f, -0.251079768f, -0.107977241f, 0.219803929f, 0.146138877f, -0.335386336f, -0.319535255f, 0.079132542f, 0.421752572f, -0.310964793f,
};

static const float32_t gru_unidirectional_float_stream_f32_update_gate_input_weights[] = {
    0.244673133f, 0.258925647f, 0.129011378f, 0.323018402f, -0.135263383f, -0.247656941f, 0.29205507f, 0.461357027f, -0.247614279f, -0.0332225f, -0.427817881f, -0.48150292f
};

static const float32_t gru_unidirectional_float_stream_f32_update_gate_hidden_weights[] = {
    -0.270374447f, 0.246330455f, -0.018747354f, -0.084194735f, 0.29524976f, -0.097509429f, 0.142206222f, -0.289820433f, -0.015031677f, 0.164797515f, 0.315074086f, 0.116467617f, 0.211315244f, 0.035253536f, 0.314859867f, -0.162634954f,
};

static const float32_t gru_unidirectional_float_stream_f32_update_gate_input_bias[] = {
    -0.205242068f, 0.175788507f, -0.111383148f, -0.0467677f
};

static const float32_t gru_unidirectional_float_stream_f32_update_gate_hidden_bias[] = {
    -0.12978901f, -0.014749109f, -0.028882211f, -0.007310458f
};

static const float32_t gru_unidirectional_float_stream_f32_reset_gate_input_weights[] = {
    0.428053439f, -0.228661045f, 0.349424243f, -0.322811872f, 0.05380442f, -0.499568701f, 0.004568026f, -0.494798005f, -0.309458613f, -0.471465975f, -0.061990783f, 0.192899361f
};

static const float32_t gru_unidirectional_float_stream_f32_reset_gate_hidden_weights[] = {
    0.427612603f, -0.124624856f, -0.238796905f, 0.260392338f, -0.241099641f, 0.241672724f, 0.412754387f, -0.276999652f, 0.300085872f, 0.3498815f, -0.356823713f, 0.268610537f, -0.186003029f, 0.121304832f, 0.087019622f, 0.436552078f,
};

static const float32_t gru_unidirectional_float_stream_f32_reset_gate_input_bias[] = {
    0.078526251f, -0.06522689f, 0.017967071f, 0.201362982f
};

static const float32_t gru_unidirectional_float_stream_f32_reset_gate_hidden_bias[] = {
    0.02618731f, 0.236048877f, 0.062268667f, 0.13499397f
};

static const float32_t gru_unidirectional_float_stream_f32_candidate_gate_input_weights[] = {
    0.407314032f, 0.010350822f, 0.301756591f, -0.018031903f, 0.181853503f, 0.227747828f, -0.480828822f, -0.08829283f, -0.340044975f, 0.37707144f, -0.147855431f, -0.08112248f
};

static const float32_t gru_unidirectional_float_stream_f32_candidate_gate_hidden_weights[] = {
    0.391169667f, 0.269873798f, 0.350463331f, -0.422784388f, 0.432592988f, 0.365510166f, -0.435722083f, -0.311197132f, -0.292197853f, 0.380004019f, 0.186238796f, -0.119350664f, 0.015433007f, -0.043899108f, -0.141465023f, -0.162049234f,
};

static const float32_t gru_unidirectional_float_stream_f32_candidate_gate_input_bias[] = {
    0.018734263f, 0.047723401f, -0.077288985f, 0.106332861f
};

static const float32_t gru_unidirectional_float_stream_f32_candidate_gate_hidden_bias[] = {
    -0.047623701f, -0.024587061f, 0.094939403f, -0.235755116f
};

#endif