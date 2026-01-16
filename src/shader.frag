precision mediump float;

uniform sampler2D tex_circuit;
uniform usampler2D tex_nets;
uniform vec2 resolution;
uniform uint target_net;

in vec2 uv;
out vec4 out_color;

void main() {
    vec2 pixel_size = 1. / resolution;

    float stroke_width = 1.;
    vec2 s = pixel_size * stroke_width;

    ivec2 tex_size = textureSize(tex_nets, 0);
    ivec2 puv = clamp(ivec2(uv * vec2(tex_size)), ivec2(0), tex_size);
    uint net = texelFetch(tex_nets, puv, 0).r;


    // 8 Neighbours
    #define coord(_x, _y) clamp(ivec2((uv + vec2(_x, _y)) * vec2(tex_size)), ivec2(0), tex_size)
    uint n0 = texelFetch(tex_nets, coord(-s.x, -s.y), 0).r;
    uint n1 = texelFetch(tex_nets, coord( 0,   -s.y), 0).r;
    uint n2 = texelFetch(tex_nets, coord( s.x, -s.y), 0).r;
    uint n3 = texelFetch(tex_nets, coord(-s.x,  0),   0).r;
    uint n4 = texelFetch(tex_nets, coord( s.x,  0),   0).r;
    uint n5 = texelFetch(tex_nets, coord(-s.x,  s.y), 0).r;
    uint n6 = texelFetch(tex_nets, coord( 0,    s.y), 0).r;
    uint n7 = texelFetch(tex_nets, coord( s.x,  s.y), 0).r;

    out_color = texture(tex_circuit, uv);

    if (target_net > 0u) {
        if (target_net == net)
            out_color = out_color * 1.3;
        else if (target_net == n0 || target_net == n1
              || target_net == n2 || target_net == n3
              || target_net == n4 || target_net == n5
              || target_net == n6 || target_net == n7)
            out_color = vec4(255);
        else if (net > 0u)
            out_color.rgb *= 0.5;
    }
}
