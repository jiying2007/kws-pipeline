#define _GNU_SOURCE
#include <fenv.h>
#include <gnu/libc-version.h>
#include <link.h>
#include <stdio.h>
#include <sys/auxv.h>

static int visit(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    (void)data;
    if (info->dlpi_name[0] != '\0') printf("OBJECT %s\n", info->dlpi_name);
    return 0;
}

int main(void) {
    unsigned int fpscr;
    __asm__ volatile ("vmrs %0, fpscr" : "=r" (fpscr));
    printf("FPSCR %08x\nROUND %d\nEXCEPT %d\nHWCAP %lu\nHWCAP2 %lu\nGLIBC %s\n",
           fpscr, fegetround(), fetestexcept(FE_ALL_EXCEPT),
           getauxval(AT_HWCAP), getauxval(AT_HWCAP2), gnu_get_libc_version());
    return dl_iterate_phdr(visit, NULL);
}
