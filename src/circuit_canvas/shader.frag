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

uint net_at(vec2 pixel) {
    vec2 tex_size = vec2(textureSize(tex_nets, 0));
    ivec2 p = ivec2(pixel);
    if (p.x < 0 || p.x >= tex_size.x || p.y < 0 || p.y >= tex_size.y)
        return 0u;
    return texelFetch(tex_nets, ivec2(p), 0).r;
}

void main() {
    out_color = vec4(0., 0., 0., 1.);

    ivec2 tex_size = textureSize(tex_nets, 0);
    vec2 tex_size_f = vec2(tex_size);

    vec2 texel_coord = uv * tex_size_f;

    float border_pixels = round(max(1., texel_size / 4.));
    // Border width in texel units
    vec2 border_size = pixel_size * border_pixels * tex_size_f;


    uint net = net_at(texel_coord);

    // Check 8 Neighbours
    // | 0 | 1 | 2 |
    // | 3 |net| 4 |
    // | 5 | 6 | 7 |
    uint n0 = net_at(texel_coord + vec2(-border_size.x, -border_size.y)); // tl
    uint n1 = net_at(texel_coord + vec2( 0,             -border_size.y)); // t
    uint n2 = net_at(texel_coord + vec2( border_size.x, -border_size.y)); // tr
    uint n3 = net_at(texel_coord + vec2(-border_size.x,  0));             // l
    uint n4 = net_at(texel_coord + vec2( border_size.x,  0));             // r
    uint n5 = net_at(texel_coord + vec2(-border_size.x,  border_size.y)); // lb
    uint n6 = net_at(texel_coord + vec2( 0,              border_size.y)); // b
    uint n7 = net_at(texel_coord + vec2( border_size.x,  border_size.y)); // rb

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
    if (inp_state != 0u && net != 1u && !is_border) {
        out_color.rgb = power_color;
        return;
    }

    // Draw gate triggered border
    if (net > wire_count && is_border) {
        uint idx = (net - wire_count) + net_count * 2u;
        uint gate_toggled = texelFetch(tex_net_state, int(idx)).r;
        if (gate_toggled == 0u) {
            vec2 inner = fract(texel_coord);
            bool border_t = fract(inner.y) < border_size.y;
            bool border_b = fract(inner.y) > 1. - border_size.y;
            bool border_l = fract(inner.x) < border_size.x;
            bool border_r = fract(inner.x) > 1. - border_size.x;

            if ((!border_l && !border_r) || (border_l && n3 == net) || (border_r && n4 == net)) {
                if (border_t) {
                    if (n1 != net && n1 > wire_count && texelFetch(tex_net_state, int(n1)).r != 0u)
                        gate_toggled = 1u;
                } else if (fract(texel_coord.y) > 1. - border_size.y) {
                    if (n6 != net && n6 > wire_count && texelFetch(tex_net_state, int(n6)).r != 0u)
                        gate_toggled = 1u;
                }
            }

            if ((!border_t && !border_b) || (border_t && n1 == net) || (border_b && n6 == net)) {
                if (fract(texel_coord.x) < border_size.x) {
                    if (n3 != net && n3 > wire_count && texelFetch(tex_net_state, int(n3)).r != 0u)
                        gate_toggled = 1u;
                } else if (fract(texel_coord.x) > 1. - border_size.x) {
                    if (n4 != net && n4 > wire_count && texelFetch(tex_net_state, int(n4)).r != 0u)
                        gate_toggled = 1u;
                }
            }

            if (gate_toggled == 0u) {
                // Draw border
                if (out_color.rgb == active_color)
                    // out_color.rgb = passive_color;
                    out_color.rgb = mix(active_color, passive_color, 0.7);
                else 
                    // out_color.rgb = active_color;
                    out_color.rgb = mix(passive_color, active_color, 0.7);
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
