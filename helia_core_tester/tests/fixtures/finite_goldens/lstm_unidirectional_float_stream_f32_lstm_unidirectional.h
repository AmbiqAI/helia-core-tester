#ifndef LSTM_UNIDIRECTIONAL_FLOAT_STREAM_F32_LSTM_UNIDIRECTIONAL_H
#define LSTM_UNIDIRECTIONAL_FLOAT_STREAM_F32_LSTM_UNIDIRECTIONAL_H

#include <stdint.h>
// Golden arrays may carry NAN/INFINITY, and this header is included ahead of any
// other translation-unit include that would define them.
#include <math.h>
#include "arm_nnfunctions.h"

static const float lstm_unidirectional_float_stream_f32_input_tensor[] = {
    -0.780473351f, -0.076173268f, -0.979150891f, 0.598568618f, 0.588701367f, 0.203874439f, -0.757103324f, 0.502486229f, -0.160482883f, 0.683489919f, -0.549579918f, 0.790302634f
};

static const float lstm_unidirectional_float_stream_f32_expected_output[] = {
    0.133278728f, 0.058342401f, 0.10917066f, 0.059413195f, -0.039022762f, -0.097628467f, -0.07599289f, -0.073883697f, 0.047617719f, -0.028640648f, -0.097779304f, 0.011414835f, -0.017671818f, -0.08620517f, -0.079264849f, -0.093903191f,
};

static const float lstm_unidirectional_float_stream_f32_forget_gate_input_weights[] = {
    -0.104124621f, 0.223164365f, 0.39167285f, -0.438417643f, -0.405057639f, -0.032587137f, -0.447348505f, -0.028300336f, -0.474359065f, 0.203085989f, 0.203564733f, 0.498580903f
};

static const float lstm_unidirectional_float_stream_f32_input_gate_input_weights[] = {
    -0.415088892f, 0.232943371f, -0.124264717f, 0.132477045f, 0.080193922f, -0.165811449f, -0.088135861f, 0.095553786f, 0.30967164f, 0.258202314f, 0.150469512f, 0.4508892f
};

static const float lstm_unidirectional_float_stream_f32_cell_gate_input_weights[] = {
    -0.360797733f, -0.22306411f, -0.132649764f, -0.357344389f, -0.057159517f, -0.11033839f, -0.175319821f, -0.451739341f, -0.34594804f, -0.483473331f, -0.20941183f, -0.061698839f
};

static const float lstm_unidirectional_float_stream_f32_output_gate_input_weights[] = {
    -0.282493681f, 0.012084557f, -0.060403436f, -0.063821726f, 0.368526667f, -0.188580036f, -0.402636588f, -0.14069435f, 0.182907507f, 0.23579815f, -0.237203434f, -0.024794379f
};

static const float lstm_unidirectional_float_stream_f32_forget_gate_hidden_weights[] = {
    -0.193971455f, 0.284766227f, 0.248663455f, -0.082672521f, -0.274779439f, -0.296155006f, -0.299941033f, 0.294569701f, -0.491452336f, 0.456865311f, -0.171207026f, 0.424319774f, -0.312334508f, 0.370285481f, 0.12552911f, 0.392545581f,
};

static const float lstm_unidirectional_float_stream_f32_input_gate_hidden_weights[] = {
    -0.308422923f, -0.369472295f, 0.140002981f, 0.463513345f, 0.459460884f, 0.337342024f, -0.461294502f, -0.448528975f, 0.18048881f, 0.377091408f, 0.084612921f, 0.402325869f, -0.447767317f, -0.276807725f, 0.191207752f, -0.013943398f,
};

static const float lstm_unidirectional_float_stream_f32_cell_gate_hidden_weights[] = {
    0.115837455f, -0.165693805f, -0.338869184f, -0.117427342f, -0.403144956f, 0.262292325f, -0.211496055f, 0.178882778f, 0.466267347f, 0.103026584f, -0.373445511f, 0.105684362f, 0.112811282f, -0.128601745f, 0.221111193f, -0.257483989f,
};

static const float lstm_unidirectional_float_stream_f32_output_gate_hidden_weights[] = {
    0.291905701f, 0.065404534f, 0.307705432f, 0.254504681f, 0.488833994f, 0.457748115f, -0.12560384f, 0.2225876f, -0.420571446f, 0.493373513f, -0.015949255f, -0.259692967f, -0.127280056f, -0.15805997f, -0.00200886f, -0.062187839f,
};

static const float lstm_unidirectional_float_stream_f32_forget_gate_bias[] = {
    -0.211878747f, -0.052039657f, 0.20650962f, 0.18699041f
};

static const float lstm_unidirectional_float_stream_f32_input_gate_bias[] = {
    0.159894511f, -0.066025801f, 0.056328215f, 0.102108665f
};

static const float lstm_unidirectional_float_stream_f32_cell_gate_bias[] = {
    -0.023325685f, -0.16026637f, -0.072542213f, -0.047944121f
};

static const float lstm_unidirectional_float_stream_f32_output_gate_bias[] = {
    -0.076597005f, -0.127822027f, 0.223839059f, -0.159131557f
};

#endif