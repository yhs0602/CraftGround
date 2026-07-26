#include "cross_gl.h"
#include <cstdio>

const char *depthConvertFragmentShader = R"(
#version 330 core
out float FragDepth;

in vec2 TexCoords;

uniform sampler2D depthTexture;
uniform float nearPlane;
uniform float farPlane;

void main() {
    float rawDepth = texture(depthTexture, TexCoords).r;

    // OpenGL Non-linear Depth -> Linear Depth
    float linearDepth = (2.0 * nearPlane * farPlane) / 
                        (farPlane + nearPlane - (2.0 * rawDepth - 1.0) * (farPlane - nearPlane));

    // Normalize by far plane (0~1)
    FragDepth = linearDepth / farPlane;
}
)";

const char *depthConvertVertexShader = R"(
#version 330 core
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = inTexCoords;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}
)";

GLuint compileShader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check the compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        printf("ERROR::SHADER::COMPILATION_FAILED\n%s\n", infoLog);
    }

    return shader;
}

GLuint
createShaderProgram(const char *vertexSource, const char *fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check the linking status
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }

    // Remove the shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

GLuint quadVAO, quadVBO;
void initQuad() {
    float quadVertices[] = {
        // Positions   // TexCoords
        -1.0f,
        1.0f,
        0.0f,
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0
    );
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float))
    );

    glBindVertexArray(0);
}

GLuint depthTexture; // Original depth texture

void initDepthTexture(int width, int height) {
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT,
        width,
        height,
        0,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        NULL
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

GLuint depthConversionShader; // Shader program for depth conversion
GLint locationNearPlane, locationFarPlane, locationDepthTexture;

GLuint depthFBO, depthRenderTexture;
void initDepthResources(int width, int height) {
    depthConversionShader = createShaderProgram(
        depthConvertVertexShader, depthConvertFragmentShader
    );

    initDepthTexture(width, height);

    // Uniform locations
    locationNearPlane =
        glGetUniformLocation(depthConversionShader, "nearPlane");
    locationFarPlane = glGetUniformLocation(depthConversionShader, "farPlane");
    locationDepthTexture =
        glGetUniformLocation(depthConversionShader, "depthTexture");

    initQuad();

    // Generate framebuffer to save the converted depth
    glGenFramebuffers(1, &depthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);

    glGenTextures(1, &depthRenderTexture);
    glBindTexture(GL_TEXTURE_2D, depthRenderTexture);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        depthRenderTexture,
        0
    );

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        printf("ERROR::FRAMEBUFFER:: Depth framebuffer is not complete!\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static float *depthPixels = nullptr;
static size_t depthPixelsSize = 0;

float *captureDepth(
    GLuint depthFramebufferId,
    int width,
    int height,
    bool requiresDepthConversion,
    float near,
    float far
) {
    glBindFramebuffer(GL_FRAMEBUFFER, depthFramebufferId);
    const size_t newDepthPixelsSize = width * height;

    if (newDepthPixelsSize != depthPixelsSize) {
        if (depthPixels != nullptr) {
            delete[] depthPixels;
        }
        depthPixels = new float[newDepthPixelsSize];
        depthPixelsSize = newDepthPixelsSize;
    }

    bool use_cpu_conversion = false;
    if (requiresDepthConversion && !use_cpu_conversion) {
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glCopyTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 0, 0, width, height, 0
        );

        // Enable the shader program
        glUseProgram(depthConversionShader);
        glUniform1f(locationNearPlane, near);
        glUniform1f(locationFarPlane, far);
        glUniform1i(locationDepthTexture, 0);

        // Bind Depth texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTexture);

        // Render the depth to the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, width, height);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, depthPixels);
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
        // Read depth pixels directly
        glReadPixels(
            0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthPixels
        );

        if (requiresDepthConversion) {
            // Convert depth values to linear space using CPU
            for (size_t i = 0; i < newDepthPixelsSize; i++) {
                float rawDepth = depthPixels[i];

                // OpenGL Non-linear Depth -> Linear Depth
                float linearDepth =
                    (2.0 * near * far) /
                    (far + near - (2.0 * rawDepth - 1.0) * (far - near));

                // Normalize by far plane (0~1)
                depthPixels[i] = linearDepth / far;
            }
        }
    }

    return depthPixels;
}
