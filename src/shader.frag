precision mediump float;

uniform sampler2D tex_circuit;
uniform usampler2D tex_nets;
uniform usamplerBuffer tex_state;

uniform vec2 pixel_size;
uniform uint target_net;

in vec2 uv;
out vec4 out_color;

void main() {
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

    uint net_state = texelFetch(tex_state, int(net)).r;

    out_color = texture(tex_circuit, uv);

    if (net_state == 0u) {
        out_color.rgb = out_color.rgb / 1.5;
    } else {
        out_color.rgb = out_color.rgb * 1.5;
    }

    if (target_net > 0u) {
        if (target_net == net) {
            ivec2 p = ivec2(gl_FragCoord);
            // if (p.x % 2 == 0 && p.y % 2 == 0) 
            if ((p.x + p.y / 2) % 2 == 0 && p.y % 2 == 0) {
            // if (p.x % 2 == p.y % 2){
                out_color.rgb += vec3(0.3);
        } else {

                out_color.rgb -= vec3(0.1);
            }
        }
        else if (target_net == n0 || target_net == n1
              || target_net == n2 || target_net == n3
              || target_net == n4 || target_net == n5
              || target_net == n6 || target_net == n7)
            out_color = vec4(255);
        // else if (net > 0u)
        //     out_color.rgb *= 0.5;
    }
}
