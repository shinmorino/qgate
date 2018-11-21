#define _CRT_SECURE_NO_WARNINGS  /* disable warning on Windows since the usage of getenv() is safe here. */
#include <stdlib.h>
#include "Types.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdexcept>



void qgate::__abort(const char *file, unsigned long line) {
    fprintf(stderr, "%s:%d %s\n", file, (int)line, "internal error, aborted.");
    abort();
}

void qgate::__abort(const char *file, unsigned long line, const char *format, ...) {
    char msg[512];

    va_list va;
    va_start(va, format);
    vsnprintf(msg, sizeof(msg), format, va);
    va_end(va);
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void qgate::__throwError(const char *file, unsigned long line) {
    __throwError(file, line, "Error");
}

void qgate::__throwError(const char *file, unsigned long line, const char *format, ...) {
    char msg[512];

    va_list va;
    va_start(va, format);
    vsnprintf(msg, sizeof(msg), format, va);
    va_end(va);
    
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s:%d %s\n", file, (int)line, msg);
    throw std::runtime_error(buffer);
}


void qgate::log(const char *format, ...) {
    static int verbose = -1;
    if (verbose == -1) {
        const char *env = getenv("SQAOD_VERBOSE");
        if ((env != NULL) && (*env != '0'))
            verbose = 1;
        else
            verbose = 0;
    }

    if (verbose) {
        va_list va;
        va_start(va, format);
        vfprintf(stderr, format, va);
        va_end(va);
        fprintf(stderr, "\n");
    }
}

