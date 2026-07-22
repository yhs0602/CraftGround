#include "cross_gl.h"
#include <cstdio>

// 16 x 16 bitmap cursor
// 0: transparent, 1: white, 2: black
// https://github.com/openai/Video-Pre-Training/blob/main/cursors/mouse_cursor_white_16x16.png
// MIT License
const GLubyte cursor[16][16] = {
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0},

    {2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0},
    {2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {2, 1, 2, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 2, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0},
};

GLuint cursorTexID;
GLuint cursorShaderProgram;
GLuint cursorVAO, cursorVBO, cursorEBO;

float cursorVertices[] = {
    // Positions      // Texture Coords
    0.0f,
    0.0f,
    0.0f,
    0.0f, // Bottom-left
    1.0f,
    0.0f,
    1.0f,
    0.0f, // Bottom-right
    1.0f,
    -1.0f,
    1.0f,
    1.0f, // Top-right
    0.0f,
    -1.0f,
    0.0f,
    1.0f // Top-left
};

// index data
unsigned int cursorIndices[] = {0, 1, 2, 2, 3, 0};

bool initCursorTexture() {
    glGenTextures(1, &cursorTexID);
    glBindTexture(GL_TEXTURE_2D, cursorTexID);

    // convert cursor to RGBA format to cursorTexture
    // 0: transparent, 1: white, 2: black
    GLubyte cursorTexture[16 * 16 * 4];

    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            int index = (y * 16 + x) * 4;
            switch (cursor[y][x]) {
            case 0:
                cursorTexture[index] = 0;
                cursorTexture[index + 1] = 0;
                cursorTexture[index + 2] = 0;
                cursorTexture[index + 3] = 0;
                break;
            case 1:
                cursorTexture[index] = 255;
                cursorTexture[index + 1] = 255;
                cursorTexture[index + 2] = 255;
                cursorTexture[index + 3] = 255;
                break;
            case 2:
                cursorTexture[index] = 0;
                cursorTexture[index + 1] = 0;
                cursorTexture[index + 2] = 0;
                cursorTexture[index + 3] = 255;
                break;
            }
        }
    }

    // Upload the cursor texture
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        16,
        16,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        cursorTexture
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    const char *vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;       // Vertex position
        layout(location = 1) in vec2 aTexCoord;  // Texture coordinates

        out vec2 TexCoord;  // Texture coordinates to fragment shader

        uniform mat4 projection;
        uniform mat4 model;

        void main() {
            gl_Position = projection * model * vec4(aPos, 0.0, 1.0); // Vertex position
            TexCoord = aTexCoord; // Pass the texture
        }
    )";

    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoord;         // Texture coordinates from vertex shader
        uniform sampler2D uTexture; // Texture sampler

        void main() {
            FragColor = texture(uTexture, TexCoord); // Output the texture
        }
    )";
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
        return false;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
        return false;
    }

    cursorShaderProgram = glCreateProgram();
    glAttachShader(cursorShaderProgram, vertexShader);
    glAttachShader(cursorShaderProgram, fragmentShader);
    glLinkProgram(cursorShaderProgram);

    glGetProgramiv(cursorShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(cursorShaderProgram, 512, nullptr, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
        return false;
    }

    // remove shaders (no longer needed after linking)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(1, &cursorVAO);
    glGenBuffers(1, &cursorVBO);
    glGenBuffers(1, &cursorEBO);

    glBindVertexArray(cursorVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cursorVBO);
    glBufferData(
        GL_ARRAY_BUFFER, sizeof(cursorVertices), cursorVertices, GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cursorEBO);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        sizeof(cursorIndices),
        cursorIndices,
        GL_STATIC_DRAW
    );

    // Position attribute (aPos)
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0
    );
    glEnableVertexAttribArray(0);

    // Texture attribute (aTexCoord)
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float))
    );
    glEnableVertexAttribArray(1);

    glBindVertexArray(0); // Unbind VAO

    return true;
}

// TODO: USE shader
/*
void renderCursor(jint mouseX, jint mouseY) {
    glBindTexture(GL_TEXTURE_2D, cursorTexID);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(mouseX, mouseY);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(mouseX + 16, mouseY);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(mouseX + 16, mouseY - 16);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(mouseX, mouseY - 16);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}
*/

void renderCursor(int mouseX, int mouseY) {
    glUseProgram(cursorShaderProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cursorTexID);
    glUniform1i(glGetUniformLocation(cursorShaderProgram, "uTexture"), 0);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(mouseX, mouseY, 0.0f));
    model = glm::scale(model, glm::vec3(16.0f, 16.0f, 1.0f));
    glUniformMatrix4fv(
        glGetUniformLocation(cursorShaderProgram, "model"),
        1,
        GL_FALSE,
        glm::value_ptr(model)
    );
    glBindVertexArray(cursorVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void drawCursorCPU(
    int xPos, int yPos, int targetSizeX, int targetSizeY, GLubyte *pixels
) {
    int cursorHeight = 16;
    int cursorWidth = 16;
    for (int dy = 0; dy < cursorHeight; ++dy) {
        for (int dx = 0; dx < cursorWidth; ++dx) {
            int pixelX = xPos + dx;
            int pixelY = yPos + dy;

            // check if the pixel is within the target image
            if (pixelX >= 0 && pixelX < targetSizeX && pixelY >= 0 &&
                pixelY < targetSizeY) {
                // Invert y axis
                int index = ((targetSizeY - pixelY) * targetSizeX + pixelX) *
                            3; // calculate the index of the pixel

                if (index >= 0 && index + 2 < targetSizeX * targetSizeY * 3) {
                    // draw black if color is 2
                    if (cursor[dy][dx] == 2) {
                        pixels[index] = 0;     // Red
                        pixels[index + 1] = 0; // Green
                        pixels[index + 2] = 0; // Blue
                    }
                    // draw white if color is 1
                    else if (cursor[dy][dx] == 1) {
                        pixels[index] = 255;     // Red
                        pixels[index + 1] = 255; // Green
                        pixels[index + 2] = 255; // Blue
                    }
                    // color is 0, do nothing (transparent)
                }
            }
        }
    }
}