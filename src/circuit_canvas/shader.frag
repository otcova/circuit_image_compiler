precision mediump float;

uniform sampler2D tex_circuit;
uniform usampler2D tex_nets;
// Contains: [...nets on/off, ...nets inputs on/off, ...gates toggled/not-toggled]
uniform usamplerBuffer tex_net_state;

uniform vec3 power_color;
uniform vec3 active_color;
uniform vec3 passive_color;
uniform float texel_size; // Size of a texel in surface pixel units

uniform vec2 pixel_size;
uniform uint selected_net;
uniform uint net_count;
uniform uint wire_count;

in vec2 uv;
out vec4 out_color;

void main() {
    out_color.a = 1.;

    float stroke_width = 1.; // (1. + texel_size / 4.); 
    stroke_width = round(max(1., texel_size / 6.));
    vec2 d = pixel_size * stroke_width; // distance to neighbour

    ivec2 tex_size = textureSize(tex_nets, 0);
    vec2 f_pix = uv * vec2(tex_size);
    ivec2 clamp_pix = clamp(ivec2(f_pix), ivec2(0), tex_size);
    uint net = texelFetch(tex_nets, clamp_pix, 0).r;

    // Check 8 Neighbours
    #define coord(_x, _y) clamp(ivec2((uv + vec2(_x, _y)) * vec2(tex_size)), ivec2(0), tex_size)
    uint n0 = texelFetch(tex_nets, coord(-d.x, -d.y), 0).r;
    uint n1 = texelFetch(tex_nets, coord( 0,   -d.y), 0).r;
    uint n2 = texelFetch(tex_nets, coord( d.x, -d.y), 0).r;
    uint n3 = texelFetch(tex_nets, coord(-d.x,  0),   0).r;
    uint n4 = texelFetch(tex_nets, coord( d.x,  0),   0).r;
    uint n5 = texelFetch(tex_nets, coord(-d.x,  d.y), 0).r;
    uint n6 = texelFetch(tex_nets, coord( 0,    d.y), 0).r;
    uint n7 = texelFetch(tex_nets, coord( d.x,  d.y), 0).r;

    uint net_state = texelFetch(tex_net_state, int(net)).r;
    uint inp_state = texelFetch(tex_net_state, int(net + net_count)).r;

    bool is_border = n0 != net || n1 != net || n2 != net || n3 != net
        || n4 != net || n5 != net || n6 != net || n7 != net;

    out_color.rgb = texture(tex_circuit, uv).rgb;

    // Draw selected net outside border
    if ((selected_net != 0u && selected_net != net)
         && (selected_net == n0 || selected_net == n1
          || selected_net == n2 || selected_net == n3
          || selected_net == n4 || selected_net == n5
          || selected_net == n6 || selected_net == n7))
    {
        uint selected_inp_state = texelFetch(tex_net_state, int(selected_net + net_count)).r;
        if (selected_inp_state == 0u || selected_net == 1u) out_color.rgb = vec3(1.);
        else out_color.rgb = power_color + vec3(0.3);
        return;
    }

    // Draw powered inputs border
    if (inp_state != 0u && net != 1u && is_border) {
        out_color.rgb = power_color;
        return;
    }

    // Draw gate is triggered
    if (net > wire_count) {
        // float x = f_pix.x * 2.;
        // float y = f_pix.y * 2.;
        // bool pattern = fract(x + y) < 0.2;//fract(x) < 0.5 && fract(y) < 0.5;
        // int x = int(gl_FragCoord.x);
        // int y = int(gl_FragCoord.y);
        // bool pattern = x % 4 == 0 && y % 4 == 0;
        bool pattern = is_border;

        if (pattern) {
            uint idx = (net - wire_count) + net_count * 2u;
            uint gate_toggled = texelFetch(tex_net_state, int(idx)).r;
            if (gate_toggled != 0u) {
                if (out_color.rgb == active_color)
                    // out_color.rgb = passive_color;
                    out_color.rgb = mix(active_color, passive_color, 0.8);
                else 
                    // out_color.rgb = active_color;
                    out_color.rgb = mix(passive_color, active_color, 0.8);
            }
        }
    }

    // Draw on/off nets
    if (net_state == 0u) {
        out_color.rgb = out_color.rgb / 1.5;
    } else {
        out_color.rgb = out_color.rgb * 1.5;
    }

}
