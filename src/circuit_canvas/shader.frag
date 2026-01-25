precision mediump float;

uniform sampler2D tex_circuit;
uniform usampler2D tex_nets;
uniform usamplerBuffer tex_state;
uniform usamplerBuffer tex_input;

uniform vec3 power_color;
uniform float texel_size; // Size of a texel in surface pixel units

uniform vec2 pixel_size;
uniform uint selected_net;

in vec2 uv;
out vec4 out_color;

void main() {
    float stroke_width = 2.;
    vec2 d = pixel_size; // distance to neighbour
    if (texel_size > 2.) d += pixel_size;

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

    uint nets = uint(textureSize(tex_state)) / 2u;
    uint net_state = texelFetch(tex_state, int(net)).r;
    uint inp_state = texelFetch(tex_state, int(net + nets)).r;

    bool is_border = n0 != net || n1 != net || n2 != net || n3 != net
        || n4 != net || n5 != net || n6 != net || n7 != net;

    // Draw powered inputs border
    if (inp_state != 0u && net != 1u && is_border) {
        out_color.rgb = power_color;
        return;
    }

    out_color = texture(tex_circuit, uv);

    // Draw on/off nets
    if (net_state == 0u) {
        out_color.rgb = out_color.rgb / 1.5;
    } else {
        out_color.rgb = out_color.rgb * 1.5;
    }

    // Continue only if a net is selected
    if (selected_net == 0u) return;

    // Draw selected net
    if (selected_net == net) {
        if (is_border) out_color.rgb += vec3(0.3);
        return;
    }

    // Draw selected net outside border
    if (selected_net == n0 || selected_net == n1
          || selected_net == n2 || selected_net == n3
          || selected_net == n4 || selected_net == n5
          || selected_net == n6 || selected_net == n7) {
        out_color = vec4(255);
        return;
    }
}
