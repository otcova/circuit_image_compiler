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
    if (pixel.x < 0 || pixel.x >= 1. || pixel.y < 0 || pixel.y >= 1.)
        return 0u;
    return texelFetch(tex_nets, ivec2(pixel * tex_size), 0).r;
}

void main() {
    out_color = vec4(0., 0., 0., 1.);

    float stroke_width = 1.; // (1. + texel_size / 4.); 
    stroke_width = round(max(1., texel_size / 4.));
    vec2 d = pixel_size * stroke_width; // distance to neighbour

    ivec2 tex_size = textureSize(tex_nets, 0);
    vec2 tex_size_f = vec2(tex_size);

    uint net = net_at(uv);

    // Check 8 Neighbours
    uint n0 = net_at((uv + vec2(-d.x, -d.y)));
    uint n1 = net_at((uv + vec2( 0,   -d.y)));
    uint n2 = net_at((uv + vec2( d.x, -d.y)));
    uint n3 = net_at((uv + vec2(-d.x,  0)));
    uint n4 = net_at((uv + vec2( d.x,  0)));
    uint n5 = net_at((uv + vec2(-d.x,  d.y)));
    uint n6 = net_at((uv + vec2( 0,    d.y)));
    uint n7 = net_at((uv + vec2( d.x,  d.y)));

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
        else out_color.rgb = power_color + vec3(0.5);
        return;
    }

    // Draw powered inputs border
    if (inp_state != 0u && net != 1u && !is_border) {
        out_color.rgb = power_color;
        return;
    }

    // Draw gate is triggered
    if (net > wire_count && is_border) {
        uint idx = (net - wire_count) + net_count * 2u;
        uint gate_toggled = texelFetch(tex_net_state, int(idx)).r;
        if (gate_toggled != 0u) {
            if (out_color.rgb == active_color)
                // out_color.rgb = passive_color;
                out_color.rgb = mix(active_color, passive_color, 0.7);
            else 
                // out_color.rgb = active_color;
                out_color.rgb = mix(passive_color, active_color, 0.7);
        }
    }

    // Draw on/off nets
    if (net_state == 0u) {
        out_color.rgb = out_color.rgb / 1.5;
    } else {
        out_color.rgb = out_color.rgb * 1.5;
    }

}
