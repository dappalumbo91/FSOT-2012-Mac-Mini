/* FSOT scalar on an Intel iGPU.
 *
 * Ivy Bridge HD 4000 stops at OpenGL 4.0, so there is no compute-shader
 * stage and no CUDA. A fragment shader over an offscreen framebuffer is the
 * graphics path this chip actually has. FP64 is not in the shader ALUs;
 * the host compares this FP32 result with the archive FP64 formula.
 *
 * Input file:  uint32 n, then n*11 little-endian float32
 *   N, P, D_eff, delta_psi, recent_hits, rho, delta_theta,
 *   scale, amplitude, trend_bias, observed (0 or 1)
 * Output file: n pairs of float32 (S, collapse code)
 *   code is 2 when S > Θ, 0 when S < -Θ, else 1. Θ = C_eff * P_var.
 * Stdout: one JSON object.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned char GLubyte;
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef float GLfloat;
typedef intptr_t GLsizeiptr;
typedef char GLchar;

typedef void *EGLDisplay;
typedef void *EGLConfig;
typedef void *EGLContext;
typedef void *EGLSurface;
typedef int32_t EGLint;
typedef unsigned int EGLBoolean;

#define EGL_FALSE 0
#define EGL_TRUE 1
#define EGL_DEFAULT_DISPLAY ((void *)0)
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_OPENGL_API 0x30A2
#define EGL_OPENGL_BIT 0x0008
#define EGL_PBUFFER_BIT 0x0001
#define EGL_SURFACE_TYPE 0x3033
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_RED_SIZE 0x3024
#define EGL_GREEN_SIZE 0x3023
#define EGL_BLUE_SIZE 0x3022
#define EGL_NONE 0x3038
#define EGL_WIDTH 0x3057
#define EGL_HEIGHT 0x3056
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001

#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_FLOAT 0x1406
#define GL_RGBA 0x1908
#define GL_NEAREST 0x2600
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_2D 0x0DE1
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_TRIANGLES 0x0004
#define GL_RGBA32F 0x8814
#define GL_R32F 0x822E
#define GL_STATIC_DRAW 0x88E4
#define GL_TEXTURE_BUFFER 0x8C2A
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_MAX_TEXTURE_SIZE 0x0D33

#define PI 3.141592653589793f
#define PHI 1.618033988749895f
#define GAMMA 0.5772156649015329f
#define ALPHA 8.082937414140405e-4f
#define PSI_CON 0.6321205588285577f
#define ETA_EFF 0.46694220692425986f
#define BETA 2.620866911333223e-17f
#define CHAOS (-0.33102418261048183f)
#define THETA_S 0.29089654054517305f
#define POOF 0.1534822148944508f
#define C_EFF 0.9577480213378242f
#define P_VAR 0.9579871226722757f
#define B_IN 0.7879784892100858f
#define A_IN 1.6668538450045732f
#define A_BLEED 1.046973630587551f
#define SUCTION 0.14703398542810284f
#define P_NEW 0.30030227667037146f
#define C_FACTOR 0.28761391128429215f
#define K 0.4201087636498879f

static void *gl_lib;
static void *egl_lib;

static EGLBoolean (*p_eglBindAPI)(EGLint);
static EGLDisplay (*p_eglGetDisplay)(void *);
static EGLBoolean (*p_eglInitialize)(EGLDisplay, EGLint *, EGLint *);
static EGLBoolean (*p_eglChooseConfig)(EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *);
static EGLContext (*p_eglCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint *);
static EGLBoolean (*p_eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
static EGLSurface (*p_eglCreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint *);
static EGLBoolean (*p_eglTerminate)(EGLDisplay);
static EGLint (*p_eglGetError)(void);

static const GLubyte *(*p_glGetString)(GLenum);
static void (*p_glGetIntegerv)(GLenum, GLint *);
static void (*p_glGenBuffers)(GLsizei, GLuint *);
static void (*p_glBindBuffer)(GLenum, GLuint);
static void (*p_glBufferData)(GLenum, GLsizeiptr, const void *, GLenum);
static void (*p_glGenTextures)(GLsizei, GLuint *);
static void (*p_glBindTexture)(GLenum, GLuint);
static void (*p_glTexBuffer)(GLenum, GLenum, GLuint);
static void (*p_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
static void (*p_glTexParameteri)(GLenum, GLenum, GLint);
static void (*p_glGenFramebuffers)(GLsizei, GLuint *);
static void (*p_glBindFramebuffer)(GLenum, GLuint);
static void (*p_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
static GLenum (*p_glCheckFramebufferStatus)(GLenum);
static void (*p_glGenVertexArrays)(GLsizei, GLuint *);
static void (*p_glBindVertexArray)(GLuint);
static GLuint (*p_glCreateShader)(GLenum);
static void (*p_glShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
static void (*p_glCompileShader)(GLuint);
static void (*p_glGetShaderiv)(GLuint, GLenum, GLint *);
static void (*p_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
static GLuint (*p_glCreateProgram)(void);
static void (*p_glAttachShader)(GLuint, GLuint);
static void (*p_glLinkProgram)(GLuint);
static void (*p_glGetProgramiv)(GLuint, GLenum, GLint *);
static void (*p_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
static void (*p_glUseProgram)(GLuint);
static GLint (*p_glGetUniformLocation)(GLuint, const GLchar *);
static void (*p_glUniform1i)(GLint, GLint);
static void (*p_glViewport)(GLint, GLint, GLsizei, GLsizei);
static void (*p_glDrawArrays)(GLenum, GLint, GLsizei);
static void (*p_glReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *);
static void (*p_glFinish)(void);

static int load_apis(void) {
    gl_lib = dlopen("libGL.so.1", RTLD_NOW);
    egl_lib = dlopen("libEGL.so.1", RTLD_NOW);
    if (!gl_lib || !egl_lib) {
        fprintf(stderr, "dlopen GL/EGL failed: %s\n", dlerror());
        return 0;
    }
#define BIND(lib, sym) do { *(void **)(&p_##sym) = dlsym((lib), #sym); if (!p_##sym) { fprintf(stderr, "missing %s\n", #sym); return 0; } } while (0)
    BIND(egl_lib, eglBindAPI);
    BIND(egl_lib, eglGetDisplay);
    BIND(egl_lib, eglInitialize);
    BIND(egl_lib, eglChooseConfig);
    BIND(egl_lib, eglCreateContext);
    BIND(egl_lib, eglMakeCurrent);
    BIND(egl_lib, eglCreatePbufferSurface);
    BIND(egl_lib, eglTerminate);
    BIND(egl_lib, eglGetError);
    BIND(gl_lib, glGetString);
    BIND(gl_lib, glGetIntegerv);
    BIND(gl_lib, glGenBuffers);
    BIND(gl_lib, glBindBuffer);
    BIND(gl_lib, glBufferData);
    BIND(gl_lib, glGenTextures);
    BIND(gl_lib, glBindTexture);
    BIND(gl_lib, glTexBuffer);
    BIND(gl_lib, glTexImage2D);
    BIND(gl_lib, glTexParameteri);
    BIND(gl_lib, glGenFramebuffers);
    BIND(gl_lib, glBindFramebuffer);
    BIND(gl_lib, glFramebufferTexture2D);
    BIND(gl_lib, glCheckFramebufferStatus);
    BIND(gl_lib, glGenVertexArrays);
    BIND(gl_lib, glBindVertexArray);
    BIND(gl_lib, glCreateShader);
    BIND(gl_lib, glShaderSource);
    BIND(gl_lib, glCompileShader);
    BIND(gl_lib, glGetShaderiv);
    BIND(gl_lib, glGetShaderInfoLog);
    BIND(gl_lib, glCreateProgram);
    BIND(gl_lib, glAttachShader);
    BIND(gl_lib, glLinkProgram);
    BIND(gl_lib, glGetProgramiv);
    BIND(gl_lib, glGetProgramInfoLog);
    BIND(gl_lib, glUseProgram);
    BIND(gl_lib, glGetUniformLocation);
    BIND(gl_lib, glUniform1i);
    BIND(gl_lib, glViewport);
    BIND(gl_lib, glDrawArrays);
    BIND(gl_lib, glReadPixels);
    BIND(gl_lib, glFinish);
#undef BIND
    return 1;
}

static float host_s(const float *a) {
    float N = a[0] < 1e-6f ? 1e-6f : a[0];
    float P = a[1];
    float D = a[2] < 1e-6f ? 1e-6f : a[2];
    float dp = a[3];
    float hits = a[4];
    float rho = a[5];
    float dth = a[6];
    float scale = a[7];
    float amp = a[8];
    float trend = a[9];
    float observed = a[10];
    float growth = expf(ALPHA * (1.f - hits / N) * GAMMA / PHI);
    float base = (N * P / sqrtf(D))
        * cosf((PSI_CON + dp) / ETA_EFF)
        * expf(-ALPHA * hits / N + rho + B_IN * dp)
        * (1.f + growth * C_EFF);
    float t1 = base * (1.f + P_NEW * logf(D / 25.f));
    if (observed > 0.5f)
        t1 *= expf(C_FACTOR * P_VAR) * cosf(dp + P_VAR);
    float t2 = scale * amp + trend;
    float valve = BETA * cosf(dp) * (N * P / sqrtf(D))
        * (1.f + CHAOS * (D - 25.f) / 25.f)
        * (1.f + POOF * cosf(THETA_S + PI) + SUCTION * sinf(THETA_S));
    float acoustic = 1.f
        + (A_BLEED * sinf(dth) * sinf(dth)) / PHI
        + (A_IN * cosf(dth) * cosf(dth)) / PHI;
    float phase = 1.f + B_IN * P_VAR;
    return K * (t1 + t2 + valve * acoustic * phase);
}

static float collapse_code(float s) {
    float theta = C_EFF * P_VAR;
    if (s > theta) return 2.f;
    if (s < -theta) return 0.f;
    return 1.f;
}

static char g_renderer[256];
static char g_version[256];
static char g_vendor[256];
static EGLDisplay g_dpy;
static int g_width;
static int g_height;

static void copy_gl_string(char *dst, size_t n, GLenum which) {
    const GLubyte *s = p_glGetString(which);
    if (!s) {
        snprintf(dst, n, "unknown");
        return;
    }
    snprintf(dst, n, "%s", (const char *)s);
}

static int context_ok(void) {
    copy_gl_string(g_renderer, sizeof g_renderer, GL_RENDERER);
    copy_gl_string(g_version, sizeof g_version, GL_VERSION);
    copy_gl_string(g_vendor, sizeof g_vendor, GL_VENDOR);
    if (strstr(g_renderer, "llvmpipe") || strstr(g_renderer, "softpipe") || strstr(g_renderer, "swrast"))
        return 0;
    return 1;
}

static int try_driver(const char *driver, int width, int height) {
    EGLint major = 0, minor = 0, ncfg = 0;
    EGLConfig cfg;
    EGLContext ctx;
    EGLSurface surf;
    const EGLint cfg_attr[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_NONE
    };
    const EGLint ctx_attr[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    const EGLint pb_attr[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };
    if (driver && driver[0])
        setenv("MESA_LOADER_DRIVER_OVERRIDE", driver, 1);
    else
        unsetenv("MESA_LOADER_DRIVER_OVERRIDE");
    if (!p_eglBindAPI(EGL_OPENGL_API))
        return 0;
    g_dpy = p_eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_dpy == NULL || !p_eglInitialize(g_dpy, &major, &minor))
        return 0;
    if (!p_eglChooseConfig(g_dpy, cfg_attr, &cfg, 1, &ncfg) || ncfg < 1)
        goto fail;
    ctx = p_eglCreateContext(g_dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (!ctx)
        goto fail;
    surf = p_eglCreatePbufferSurface(g_dpy, cfg, pb_attr);
    if (!surf || !p_eglMakeCurrent(g_dpy, surf, surf, ctx))
        goto fail;
    if (!context_ok())
        goto fail;
    g_width = width;
    g_height = height;
    return 1;
fail:
    p_eglTerminate(g_dpy);
    g_dpy = NULL;
    return 0;
}

static const char *VS =
    "#version 330 core\n"
    "void main() {\n"
    "  const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));\n"
    "  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);\n"
    "}\n";

static const char *FS =
    "#version 330 core\n"
    "uniform samplerBuffer inputs;\n"
    "uniform int width;\n"
    "uniform int n;\n"
    "out vec4 frag;\n"
    "void main() {\n"
    "  int i = int(gl_FragCoord.y) * width + int(gl_FragCoord.x);\n"
    "  if (i < 0 || i >= n) { frag = vec4(0.0); return; }\n"
    "  int b = i * 11;\n"
    "  float N = max(texelFetch(inputs, b + 0).r, 1e-6);\n"
    "  float P = texelFetch(inputs, b + 1).r;\n"
    "  float D = max(texelFetch(inputs, b + 2).r, 1e-6);\n"
    "  float dp = texelFetch(inputs, b + 3).r;\n"
    "  float hits = texelFetch(inputs, b + 4).r;\n"
    "  float rho = texelFetch(inputs, b + 5).r;\n"
    "  float dth = texelFetch(inputs, b + 6).r;\n"
    "  float scale = texelFetch(inputs, b + 7).r;\n"
    "  float amp = texelFetch(inputs, b + 8).r;\n"
    "  float trend = texelFetch(inputs, b + 9).r;\n"
    "  float observed = texelFetch(inputs, b + 10).r;\n"
    "  const float PI = 3.141592653589793;\n"
    "  const float PHI = 1.618033988749895;\n"
    "  const float GAMMA = 0.5772156649015329;\n"
    "  const float ALPHA = 8.082937414140405e-4;\n"
    "  const float PSI_CON = 0.6321205588285577;\n"
    "  const float ETA_EFF = 0.46694220692425986;\n"
    "  const float BETA = 2.620866911333223e-17;\n"
    "  const float CHAOS = -0.33102418261048183;\n"
    "  const float THETA_S = 0.29089654054517305;\n"
    "  const float POOF = 0.1534822148944508;\n"
    "  const float C_EFF = 0.9577480213378242;\n"
    "  const float P_VAR = 0.9579871226722757;\n"
    "  const float B_IN = 0.7879784892100858;\n"
    "  const float A_IN = 1.6668538450045732;\n"
    "  const float A_BLEED = 1.046973630587551;\n"
    "  const float SUCTION = 0.14703398542810284;\n"
    "  const float P_NEW = 0.30030227667037146;\n"
    "  const float C_FACTOR = 0.28761391128429215;\n"
    "  const float K = 0.4201087636498879;\n"
    "  float growth = exp(ALPHA * (1.0 - hits / N) * GAMMA / PHI);\n"
    "  float base = (N * P / sqrt(D))\n"
    "    * cos((PSI_CON + dp) / ETA_EFF)\n"
    "    * exp(-ALPHA * hits / N + rho + B_IN * dp)\n"
    "    * (1.0 + growth * C_EFF);\n"
    "  float t1 = base * (1.0 + P_NEW * log(D / 25.0));\n"
    "  float quirk = exp(C_FACTOR * P_VAR) * cos(dp + P_VAR);\n"
    "  t1 *= mix(1.0, quirk, step(0.5, observed));\n"
    "  float t2 = scale * amp + trend;\n"
    "  float valve = BETA * cos(dp) * (N * P / sqrt(D))\n"
    "    * (1.0 + CHAOS * (D - 25.0) / 25.0)\n"
    "    * (1.0 + POOF * cos(THETA_S + PI) + SUCTION * sin(THETA_S));\n"
    "  float acoustic = 1.0\n"
    "    + (A_BLEED * sin(dth) * sin(dth)) / PHI\n"
    "    + (A_IN * cos(dth) * cos(dth)) / PHI;\n"
    "  float phase = 1.0 + B_IN * P_VAR;\n"
    "  float S = K * (t1 + t2 + valve * acoustic * phase);\n"
    "  float theta = C_EFF * P_VAR;\n"
    "  float code = 1.0;\n"
    "  if (S > theta) code = 2.0;\n"
    "  else if (S < -theta) code = 0.0;\n"
    "  frag = vec4(S, code, 0.0, 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint sh = p_glCreateShader(type);
    p_glShaderSource(sh, 1, &src, NULL);
    p_glCompileShader(sh);
    GLint ok = 0;
    p_glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        p_glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
        return 0;
    }
    return sh;
}

static void json_escape(FILE *fp, const char *s) {
    for (; *s; s++) {
        if (*s == '"' || *s == '\\')
            fputc('\\', fp);
        if (*s == '\n') {
            fputs("\\n", fp);
            continue;
        }
        fputc(*s, fp);
    }
}

static void fail_json(const char *msg) {
    printf("{\"ok\":false,\"backend\":\"intel-igpu-gl330\",\"error\":\"");
    json_escape(stdout, msg);
    printf("\"}\n");
}

/* side x side field, repeated draws. Throughput is the draw+finish interval.
   Readback is timed once, after the timed draws. */
int main(int argc, char **argv) {
    int side = argc > 1 ? atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 30;
    if (side < 8 || side > 2048 || reps < 1 || reps > 10000) {
        fail_json("side must be 8..2048 and reps 1..10000");
        return 2;
    }
    if (!load_apis()) {
        fail_json("OpenGL or EGL library missing");
        return 1;
    }
    uint32_t n = (uint32_t)side * (uint32_t)side;
    float *samples = calloc((size_t)n * 11, sizeof(float));
    if (!samples) {
        fail_json("out of memory");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        float luma = 0.15f + 0.85f * (float)(i % 997) / 996.f;
        float *a = samples + (size_t)i * 11;
        a[0] = luma * 4.f;
        a[1] = luma;
        a[2] = 25.f;
        a[3] = (float)(i % 64) / 63.f * 6.283185307179586f;
        a[4] = luma * 0.3f;
        a[5] = 1.f;
        a[6] = 1.f;
        a[7] = luma;
        a[8] = 1.f;
        a[9] = 0.f;
        a[10] = 0.f;
    }
    if (!try_driver("crocus", side, side) && !try_driver("", side, side)) {
        free(samples);
        fail_json("no hardware OpenGL context");
        return 1;
    }
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS);
    if (!vs || !fs) {
        free(samples);
        fail_json("shader compile failed");
        return 1;
    }
    GLuint prog = p_glCreateProgram();
    p_glAttachShader(prog, vs);
    p_glAttachShader(prog, fs);
    p_glLinkProgram(prog);
    GLint linked = 0;
    p_glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        free(samples);
        fail_json("shader link failed");
        return 1;
    }
    GLuint vao = 0, tbo = 0, ttex = 0, color = 0, fbo = 0;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &tbo);
    p_glBindBuffer(GL_TEXTURE_BUFFER, tbo);
    p_glBufferData(GL_TEXTURE_BUFFER, (GLsizeiptr)((size_t)n * 11 * sizeof(float)), samples, GL_STATIC_DRAW);
    p_glGenTextures(1, &ttex);
    p_glBindTexture(GL_TEXTURE_BUFFER, ttex);
    p_glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo);
    p_glGenTextures(1, &color);
    p_glBindTexture(GL_TEXTURE_2D, color);
    p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, side, side, 0, GL_RGBA, GL_FLOAT, NULL);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    p_glGenFramebuffers(1, &fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    int fp16 = 0;
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        p_glTexImage2D(GL_TEXTURE_2D, 0, 0x881A, side, side, 0, GL_RGBA, GL_FLOAT, NULL);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        fp16 = 1;
    }
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        free(samples);
        fail_json("floating-point framebuffer incomplete");
        return 1;
    }
    p_glUseProgram(prog);
    p_glUniform1i(p_glGetUniformLocation(prog, "inputs"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog, "width"), side);
    p_glUniform1i(p_glGetUniformLocation(prog, "n"), (int)n);
    p_glViewport(0, 0, side, side);

    p_glDrawArrays(GL_TRIANGLES, 0, 3);
    p_glFinish();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++)
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    p_glFinish();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double draw_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    float *pixels = calloc((size_t)n * 4, sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    p_glReadPixels(0, 0, side, side, GL_RGBA, GL_FLOAT, pixels);
    p_glFinish();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double read_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    double max_abs = 0.0;
    int code_mismatch = 0;
    /* Spot-check every 64th sample against the C FP32 formula. */
    uint32_t checked = 0;
    for (uint32_t i = 0; i < n; i += 64) {
        float s = pixels[i * 4];
        float href = host_s(samples + (size_t)i * 11);
        float ad = fabsf(s - href);
        if (ad > max_abs)
            max_abs = ad;
        if (pixels[i * 4 + 1] != collapse_code(s))
            code_mismatch++;
        checked++;
    }
    double draw_s = draw_ms / 1e3;
    double per_s = draw_s > 0 ? ((double)n * reps) / draw_s : 0;
    int cur = 0, act = 0;
    FILE *f = fopen("/sys/class/drm/card1/gt_cur_freq_mhz", "r");
    if (f) { fscanf(f, "%d", &cur); fclose(f); }
    f = fopen("/sys/class/drm/card1/gt_act_freq_mhz", "r");
    if (f) { fscanf(f, "%d", &act); fclose(f); }

    printf("{\"ok\":true,\"backend\":\"intel-igpu-gl330\",\"renderer\":\"");
    json_escape(stdout, g_renderer);
    printf("\",\"gl\":\"");
    json_escape(stdout, g_version);
    printf("\",\"side\":%d,\"samples\":%u,\"reps\":%d,\"fp16_target\":%d,"
           "\"draw_ms\":%.3f,\"readback_ms\":%.3f,\"scalar_per_s\":%.3f,"
           "\"checked\":%u,\"max_abs_err_vs_fp32\":%.3e,\"code_mismatch\":%d,"
           "\"gt_cur_mhz\":%d,\"gt_act_mhz\":%d}\n",
           side, n, reps, fp16, draw_ms, read_ms, per_s, checked, max_abs, code_mismatch, cur, act);
    free(pixels);
    free(samples);
    if (g_dpy)
        p_eglTerminate(g_dpy);
    return 0;
}

