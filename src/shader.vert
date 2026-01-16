out vec2 uv;

void main() {
    vec2 pos = -1. + vec2(
        float((gl_VertexID & 1) << 2),
        float((gl_VertexID & 2) << 1)
    );
    uv = vec2(pos.x, -pos.y) * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
