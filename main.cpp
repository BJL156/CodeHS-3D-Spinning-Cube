#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#define PI 3.14159265358979323846

double clamp(double d, double min, double max) {
    return std::max(min, std::min(d, max));
}

struct Vec3 {
    Vec3();
    Vec3(double x);
    Vec3(double x, double y, double z);
    ~Vec3() = default;
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3::Vec3()
    : x(0.0), y(0.0), z(0.0) {
    
}

Vec3::Vec3(double x)
    : x(x), y(x), z(x) {
    
}

Vec3::Vec3(double x, double y, double z)
    : x(x), y(y), z(z) {
    
}

Vec3 operator+(const Vec3& v0, const Vec3& v1) {
    return { v0.x + v1.x, v0.y + v1.y, v0.z + v1.z };
}

Vec3 operator-(const Vec3& v0, const Vec3& v1) {
    return { v0.x - v1.x, v0.y - v1.y, v0.z - v1.z };
}

Vec3 operator*(const Vec3& v0, const Vec3& v1) {
    return { v0.x * v1.x, v0.y * v1.y, v0.z * v1.z };
}

Vec3 operator/(const Vec3& v0, const Vec3& v1) {
    if (v1.x == 0 || v1.y == 0 || v1.z == 0) {
        throw std::runtime_error("Can't divide by zero.");
    }
    
    return { v0.x / v1.x, v0.y / v1.y, v0.z / v1.z };
}

Vec3 operator+(double t, const Vec3& v0) {
    return { t + v0.x, t + v0.y, t + v0.z };
}

Vec3 operator-(double t, const Vec3& v0) {
    return { t - v0.x, t - v0.y, t - v0.z };
}

Vec3 operator*(double t, const Vec3& v0) {
    return { t * v0.x, t * v0.y, t * v0.z };
}

Vec3 operator/(double t, const Vec3& v0) {
    if (v0.x == 0 || v0.y == 0 || v0.z == 0) {
        throw std::runtime_error("Can't divide by zero.");
    }
    
    return { t / v0.x, t / v0.y, t / v0.z };
}

Vec3 operator+(const Vec3& v0, double t) {
    return { v0.x + t, v0.y + t, v0.z + t };
}

Vec3 operator-(const Vec3& v0, double t) {
    return { v0.x - t, v0.y - t, v0.z - t };
}

Vec3 operator*(const Vec3& v0, double t) {
    return { v0.x * t, v0.y * t, v0.z * t };
}

Vec3 operator/(const Vec3& v0, double t) {
    if (t == 0) {
        throw std::runtime_error("Can't divide by zero");
    }
    
    return { v0.x / t, v0.y / t, v0.z / t };
}

Vec3 operator-(const Vec3& v0) {
    return { -v0.x, -v0.y, -v0.z };
}

double dot(const Vec3& v0, const Vec3& v1) {
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

double lengthSquared(const Vec3& v0) {
    return v0.x*v0.x + v0.y*v0.y + v0.z*v0.z;
}

double length(const Vec3& v0) {
    return std::sqrt(lengthSquared(v0));
}

Vec3 normalize(const Vec3& v0) {
    double len = length(v0);
    if (len > 0.0) {
        return v0 / len;
    }
    
    return v0;
}

struct Mat4 {
    Mat4();
    ~Mat4() = default;
    
    double* operator[](std::size_t i);
    const double* operator[](std::size_t i) const;
    
    double data[4][4];
};

Mat4::Mat4() {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

double* Mat4::operator[](std::size_t i) {
    return data[i];
}

const double* Mat4::operator[](std::size_t i) const {
    return data[i];
}

Mat4 operator*(const Mat4& m0, const Mat4& m1) {
    Mat4 result{};
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = 0.0;
            
            for (int k = 0; k < 4; ++k) {
                result[i][j] += m0[i][k] * m1[k][j];
            }
        }
    }
    
    return result;
}

Vec3 operator*(const Mat4& m0, const Vec3& v0) {
    double x = v0.x;
    double y = v0.y;
    double z = v0.z; 
    double w = 1.0;
    
    return {
        m0[0][0] * x + m0[0][1] * y + m0[0][2] * z + m0[0][3] * w,
        m0[1][0] * x + m0[1][1] * y + m0[1][2] * z + m0[1][3] * w,
        m0[2][0] * x + m0[2][1] * y + m0[2][2] * z + m0[2][3] * w,
    };
}

Mat4 translate(const Vec3& v0) {
    Mat4 result{};
    
    result[0][3] = v0.x;
    result[1][3] = v0.y;
    result[2][3] = v0.z;
    
    return result;
}

Mat4 scale(const Vec3& v0) {
    Mat4 result{};
    
    result[0][0] = v0.x;
    result[1][1] = v0.y;
    result[2][2] = v0.z;
    
    return result;
}

Mat4 rotateX(double angle) {
    Mat4 result{};
    
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    
    result[1][1] = cosAngle;
    result[2][1] = -sinAngle;
    result[1][2] = sinAngle;
    result[2][2] = cosAngle;
    
    return result;
}

Mat4 rotateY(double angle) {
    Mat4 result{};
    
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    
    result[0][0] = cosAngle;
    result[0][2] = -sinAngle;
    result[2][0] = sinAngle;
    result[2][2] = cosAngle;
    
    return result;
}

Mat4 rotateZ(double angle) {
    Mat4 result{};
    
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    
    result[0][0] = cosAngle;
    result[0][1] = -sinAngle;
    result[1][0] = sinAngle;
    result[1][1] = cosAngle;
    
    return result;
}

Mat4 perspective(double fov, double aspectRatio, double near, double far) {
    Mat4 result{};
    
    double tanHalfFov = std::tan(fov / 2.0);
    result[0][0] = 1.0 / (aspectRatio * tanHalfFov);
    result[1][1] = 1.0 / tanHalfFov;
    result[2][2] = far / (far - near);
    result[2][3] = (-far * near) / (far - near);
    result[3][2] = 1.0;
    
    return result;
}

double getRadians(double degrees) {
    return degrees * (PI / 180.0);
}

double getCalculatedHeight(int width, double aspectRatio) {
    const double consolePixelWidthtoHeight = 2.0;
    return width / (aspectRatio * 2.0);
}

class Framebuffer {
public:
    Framebuffer(int width, double aspectRatio);
    ~Framebuffer() = default;
    
    std::string getAnsiString(const Vec3& color);
    void present();
    void clearConsole();
    
    void setPixel(int x, int y, const Vec3& color, double depth);
    void clear();
    
    int getWidth() const;
    int getHeight() const;
    double getAspectRatio() const;
private:
    void initialize();
    
    int m_width;
    int m_height;
    double m_aspectRatio;
    std::vector<Vec3> m_framebuffer;
    std::vector<double> m_depthBuffer;
    Vec3 m_defaultColor = { 1.0, 1.0, 1.0 };
};

Framebuffer::Framebuffer(int width, double aspectRatio)
    : m_width(width), m_height(getCalculatedHeight(width, aspectRatio)), m_aspectRatio(aspectRatio) {
    initialize();
}

std::string Framebuffer::getAnsiString(const Vec3& color) {
    Vec3 scaledColor = {
        clamp(color.x * 255.0, 0.0, 255.0),
        clamp(color.y * 255.0, 0.0, 255.0),
        clamp(color.z * 255.0, 0.0, 255.0)
    };
    
    std::string result = "\033[48;2;" +
        std::to_string(static_cast<int>(scaledColor.x)) + ";" +
        std::to_string(static_cast<int>(scaledColor.y)) + ";" +
        std::to_string(static_cast<int>(scaledColor.z)) + "m";
    
    return result;
}

void Framebuffer::present() {
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            Vec3 pixelColor = m_framebuffer[y * m_width + x];
            std::cout << getAnsiString(pixelColor) << ' ';
        }
        
        std::cout << "\033[0m\n";
    }
}

void Framebuffer::clearConsole() {
    std::cout << "\033[2J\033[1;1H";
}

void Framebuffer::setPixel(int x, int y, const Vec3& color, double depth) {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) {
        throw std::out_of_range("Pixel coordinates are out of range.\n");
    }
    
    int index = y * m_width + x;
    
    if (depth < m_depthBuffer[index]) {
        m_framebuffer[index] = color;
        m_depthBuffer[index] = depth;
    }
}

void Framebuffer::clear() {
    std::fill(m_framebuffer.begin(), m_framebuffer.end(), m_defaultColor);
    std::fill(m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits<double>::infinity());
}

int Framebuffer::getWidth() const {
    return m_width;
}

int Framebuffer::getHeight() const {
    return m_height;
}

double Framebuffer::getAspectRatio() const {
    return m_aspectRatio;
}

void Framebuffer::initialize() {
    int amountOfPixels = m_width*m_height;
    m_framebuffer.resize(amountOfPixels);
    m_depthBuffer.resize(amountOfPixels, std::numeric_limits<double>::infinity());
}

Vec3 ndcToScreenSpace(const Vec3& ndc, const Framebuffer& framebuffer) {
    return {
        (ndc.x + 1.0) * 0.5 * framebuffer.getWidth(),
        (1.0 - ndc.y) * 0.5 * framebuffer.getHeight(),
        0.0
    };
}

Vec3 getBarycentricCoords(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    double det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    double factorAlpha = (b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y);
    double factorBeta = (c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y);
    double alpha =  factorAlpha / det;
    double beta =  factorBeta / det;
    
    double gamma = 1.0 - alpha - beta;
    
    return { alpha, beta, gamma };
}

Vec3 getInterpolation(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& barycentric) {
    return {
        a.x * barycentric.x + b.x * barycentric.y + c.x * barycentric.z,
        a.y * barycentric.x + b.y * barycentric.y + c.y * barycentric.z,
        a.z * barycentric.x + b.z * barycentric.y + c.z * barycentric.z
    };
}

bool isPointInTriangle(const Vec3& currentScreenSpace, const std::vector<Vec3> screenSpaceTriangle, const Vec3 barycentric) {
    return barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0;
}

void drawTriangle(const std::vector<Vec3>& triangle, Framebuffer& framebuffer) {
    std::vector<Vec3> screenSpaceTriangle;
    for (const Vec3& ndc : triangle) {
        screenSpaceTriangle.push_back(ndcToScreenSpace(ndc, framebuffer));
    }
    
    int width = framebuffer.getWidth();
    int height = framebuffer.getHeight();
    
    int minX = static_cast<int>(clamp(std::floor(std::min({ screenSpaceTriangle[0].x, screenSpaceTriangle[1].x, screenSpaceTriangle[2].x })), 0.0, width - 1));
    int maxX = static_cast<int>(clamp(std::ceil(std::max({ screenSpaceTriangle[0].x, screenSpaceTriangle[1].x, screenSpaceTriangle[2].x })), 0.0, width - 1));
    int minY = static_cast<int>(clamp(std::floor(std::min({ screenSpaceTriangle[0].y, screenSpaceTriangle[1].y, screenSpaceTriangle[2].y })), 0.0, height - 1));
    int maxY = static_cast<int>(clamp(std::ceil(std::max({ screenSpaceTriangle[0].y, screenSpaceTriangle[1].y, screenSpaceTriangle[2].y })), 0.0, height - 1));
    
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Vec3 currentScreenSpace = { (double)x, (double)y, 0 };
            
            Vec3 barycentric = getBarycentricCoords(currentScreenSpace, screenSpaceTriangle[0], screenSpaceTriangle[1], screenSpaceTriangle[2]);
            if (isPointInTriangle(currentScreenSpace, screenSpaceTriangle, barycentric)) {
                Vec3 interpolatedColor = getInterpolation({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, barycentric);
                
                double depth = getInterpolation(
                    triangle[0].z,
                    triangle[1].z,
                    triangle[2].z,
                    barycentric
                ).z;
                
                framebuffer.setPixel(x, y, interpolatedColor, depth);
            }
        }
    }
}

Mat4 getTransform(const Vec3& size, const Vec3& rotation, const Vec3& translation, double fov, double aspectRatio, double near, double far) {
    Mat4 scaleMat = scale(size);
    Mat4 rotationMat = rotateZ(rotation.z) * rotateY(rotation.y) * rotateX(rotation.x);
    Mat4 translationMat = translate(translation);
    Mat4 modelMat = translationMat * rotationMat * scaleMat;
    
    Mat4 projectionMat = perspective(fov, aspectRatio, near, far);
    
    return projectionMat * modelMat;
}

double getDeltaTime() {
    static auto previousFrameTime = std::chrono::high_resolution_clock::now();
    
    auto currentFrameTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaTime = currentFrameTime - previousFrameTime;
    previousFrameTime = currentFrameTime;
    
    return deltaTime.count();
}

class Application {
public:
    Application() = default;
    ~Application() = default;
    
    void run();
private:
    void presentFrame();
    
    const int m_imageWidth = 128;
    const double m_fov = 45.0;
    const double m_aspectRatio = 16.0 / 9.0;
    const double m_near = 0.1;
    const double m_far = 100.0;
    Framebuffer m_framebuffer{ m_imageWidth, m_aspectRatio };
};

void Application::run() {
    std::vector<Vec3> cubeVertices = {
        { -0.5, -0.5,  0.5 },
        {  0.5, -0.5,  0.5 },
        {  0.5,  0.5,  0.5 },
        { -0.5,  0.5,  0.5 },
        { -0.5, -0.5, -0.5 },
        {  0.5, -0.5, -0.5 },
        {  0.5,  0.5, -0.5 },
        { -0.5,  0.5, -0.5 }
    };
    
    std::vector<std::vector<int>> cubeTriangles = {
        { 0, 1, 2 }, { 2, 3, 0 },
        { 4, 5, 6 }, { 6, 7, 4 },
        { 0, 4, 7 }, { 7, 3, 0 },
        { 1, 5, 6 }, { 6, 2, 1 },
        { 0, 1, 5 }, { 5, 4, 0 },
        { 3, 2, 6 }, { 6, 7, 3 }
    };
    
    double rotationSpeed = 50.0;
    double angle = 0.0;
    
    while (true) {
        double deltaTime = getDeltaTime();
        
        angle += rotationSpeed * deltaTime;
        
        Mat4 transformMat = getTransform(
            { 1.0, 1.0, 1.0 },
            { getRadians(angle), getRadians(angle), 0.0 },
            { 0.0, 0.0, 2.0 },
            getRadians(m_fov),
            m_aspectRatio,
            m_near,
            m_far
        );
        
        m_framebuffer.clear();
        
        for (const auto& triangle : cubeTriangles) {
            std::vector<Vec3> transformedTriangle;
            
            for (int i = 0; i < 3; ++i) {
                Vec3 vertex = cubeVertices[triangle[i]];
                Vec3 transformedVertex = transformMat * vertex;
                
                double w = transformMat[3][0] * vertex.x + transformMat[3][1] * vertex.y + transformMat[3][2] * vertex.z + transformMat[3][3];
                if (w == 0.0) {
                    continue;
                }
                
                transformedVertex.x = transformedVertex.x / w;
                transformedVertex.y = transformedVertex.y / w;
                transformedVertex.z = transformedVertex.z / w;
                
                transformedTriangle.push_back(transformedVertex);
            }
            
            drawTriangle(transformedTriangle, m_framebuffer);
        }
        
        presentFrame();
        
        std::cin.get();
    }
}

void Application::presentFrame() {
    m_framebuffer.clearConsole();
    m_framebuffer.present();
}

int main(void) {
    Application app{};
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
