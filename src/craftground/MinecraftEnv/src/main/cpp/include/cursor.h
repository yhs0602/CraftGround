#pragma once
#include "cross_gl.h"
void drawCursorCPU(
    int xPos, int yPos, int targetSizeX, int targetSizeY, GLubyte *pixels
);
void renderCursor(int mouseX, int mouseY);
bool initCursorTexture();