#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*extract_proc_t)(const char *shared_cache_file_path,
                              const char *extraction_root_path,
                              void (^progress)(unsigned current, unsigned total));

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <dyld_shared_cache> <output_dir>\n", argv[0]);
        return 2;
    }

    void *handle = dlopen("/usr/lib/dsc_extractor.bundle", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 3;
    }

    extract_proc_t extract = (extract_proc_t)dlsym(handle, "dyld_shared_cache_extract_dylibs_progress");
    if (!extract) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 4;
    }

    int result = extract(argv[1], argv[2], ^(unsigned current, unsigned total) {
        if (total == 0 || current == total || current % 200 == 0) {
            fprintf(stderr, "extract %u/%u\n", current, total);
        }
    });
    fprintf(stderr, "extract result=%d\n", result);
    return result;
}
