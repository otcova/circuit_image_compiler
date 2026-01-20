uniform vec2 uv_min;
uniform vec2 uv_size;

out vec2 uv;

void main() {
    vec2 pos = -1. + vec2(
        float((gl_VertexID & 1) << 2),
        float((gl_VertexID & 2) << 1)
    );
    // Scale from 0 to 1
    uv = vec2(pos.x, -pos.y) * 0.5 + 0.5;
    // Scale from uv_min to uv_min + uv_size
    uv = uv_min + uv * uv_size;

    gl_Position = vec4(pos, 0.0, 1.0);
}
