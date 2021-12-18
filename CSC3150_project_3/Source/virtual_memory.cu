#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int ReplacePage(VirtualMemory* vm, int VPN);
__device__ int PageIndex2ByteOffset(int pageIndex);
__device__ int FindLRUPageIndex(VirtualMemory* vm);

/*
* Initialize the page table
* page: 32 bytes, entry: 1024
* we need to compare the current VPN with the VPN of each page entry
*/
__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // valid bit: MSB (1: invlaid), dirty bit: LSB (1: dirty)
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

/*
* Initialize the virtual memory space
*/
__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage,
    u32* invert_page_table, int* pagefault_num_ptr, u32* ptCounter,
    int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
    int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
    int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->pageTableCounter = ptCounter;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

/*
* read data from buffer (physical memory)
* -----------------------------------------
* When reading data from vm_buffer, if such page exits in share memory, read it out
* Otherwise, replace the LRU set. Swap the LRU page out of share memory and swap it in secondary storage
* Swap the designed page into share memory for data reading & update page table
*/
__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
    /* read single element from data buffer */
    /* to read single element from data buffer */
    u32 VPN = addr >> 5; // logic address: VPN
    u32 offset = addr & 0b11111; // logic address: offset

    int size = vm->PAGE_ENTRIES;
    for (int i = 0; i < size; i++) { // traverse the page entries to find the matching VPN
        if (vm->invert_page_table[i + size] == VPN) { // if there is a match
            if (vm->invert_page_table[i] >> 31) { // invalid
                *(vm->pagefault_num_ptr) += 1;

                // load the page from disk to physical memory
                for (int j = 0; j < 32; j++) {
                    (vm->buffer + PageIndex2ByteOffset(i))[j] = (vm->storage + PageIndex2ByteOffset(VPN))[j];
                }
                // set dirty bit: clean
                vm->invert_page_table[i] &= 0xFFFFFFFE;

                // update the page table: set valid bit
                vm->invert_page_table[i] &= 0x7FFFFFFF;

                // update the page table counter
                for (int k = 0; k < size; k++) { // increment counter
                    vm->pageTableCounter[k] += 1;
                }
                vm->pageTableCounter[i] = 0; // reset counter of the accessed page

                // read from physical memory
                return (vm->buffer + PageIndex2ByteOffset(i))[offset];
            }
            else { // valid
                // update the page table counter
                for (int k = 0; k < size; k++) { // increment counter
                    vm->pageTableCounter[k] += 1;
                }
                vm->pageTableCounter[i] = 0; // reset counter of the accessed page

                // do not modify the dirty bit

                // directly read from the page, no need to update page table
                return (vm->buffer + PageIndex2ByteOffset(i))[offset];
            }
        }
    }
    // the VPN is not in the page entries
    // page replacement (update the page table)
    int VPNPageIndex = ReplacePage(vm, VPN);

    // update the page table counter
    for (int k = 0; k < size; k++) { // increment counter
        vm->pageTableCounter[k] += 1;
    }
    vm->pageTableCounter[VPNPageIndex] = 0; // reset counter of the accessed page

    // do not change the dirty bit

    // read the page
    return (vm->buffer + PageIndex2ByteOffset(VPNPageIndex))[offset];
}

/*
* write data to buffer (physical memory)
* -----------------------------------------
* When writing data to vm_buffer, if share memory is available, place data to available page.
* Otherwise, replace the LRU set. Swap the LRU page out of share memory and swap it in secondary storage.
* Swap the designed page into share memory for data reading & update page table
*/
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
    /* write value into data buffer */
    u32 VPN = addr >> 5; // logic address: VPN
    u32 offset = addr & 0b11111; // logic address: offset

    int size = vm->PAGE_ENTRIES;
    for (int i=0; i < size; i++) { // traverse the page table to find matching VPN
        if (vm->invert_page_table[i + size] == VPN) { // there is matching VPN
            if (vm->invert_page_table[i]>>31) {// invalid
                *(vm->pagefault_num_ptr) += 1;

                // load the desired page from disk to the physical memory
                for (int j = 0; j < 32; j++) {
                    (vm->buffer + PageIndex2ByteOffset(i))[j] = (vm->storage + PageIndex2ByteOffset(VPN))[j];
                }
                // set dirty bit: dirty
                vm->invert_page_table[i] |= 0x1;

                // update page table: set valid bit
                vm->invert_page_table[i] &= 0x7FFFFFFF;

                // update the page table counter
                for (int k = 0; k < size; k++) { // increment counter
                    vm->pageTableCounter[k] += 1;
                }
                vm->pageTableCounter[i] = 0; // reset counter of the accessed page

                // write to the physical memory
                (vm->buffer + PageIndex2ByteOffset(i))[offset] = value;
                return;
            }
            else {// valid
                // update the page table counter
                for (int k = 0; k < size; k++) { // increment counter
                    vm->pageTableCounter[k] += 1;
                }
                vm->pageTableCounter[i] = 0; // reset counter of the accessed page

                // set dirty bit: dirty
                vm->invert_page_table[i] |= 0x1;

                // directly write to the physical memory, no need to update page table
                (vm->buffer + PageIndex2ByteOffset(i))[offset] = value;
                return;
            }
        }
    }
    // the VPN not in the page entries
    // page replacement (update the page table)
    int VPNPageIndex = ReplacePage(vm, VPN);

    // update the page table counter
    for (int k = 0; k < size; k++) { // increment counter
        vm->pageTableCounter[k] += 1;
    }
    vm->pageTableCounter[VPNPageIndex] = 0; // reset counter of the accessed page

    // set dirty bit: dirty
    vm->invert_page_table[VPNPageIndex] |= 0x1;

    // write the page in the physical memory
    (vm->buffer + PageIndex2ByteOffset(VPNPageIndex))[offset] = value;
    return;
}

/*
* load elements of buffer (phycial memory) to results buffer (global memory)
* -----------------------------------------
* use LRU algorithm, LRU page to output first
*/
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
    /* snapshot function togther with vm_read to load elements from data */
    for (int i = 0; i < input_size; i++) {
        results[i + offset] = vm_read(vm, i);
    }

}

/*
* Convert: page index -> byte index
* a page is data transfer unit, while a byte is memory unit
* map data transfer unit with byte memory unit
*/
__device__ int PageIndex2ByteOffset(int pageIndex) {
    return pageIndex * 32;
}

/* 
* perform page replacement
* 1. find LRU page
* 2. sawp the LRU page out, swap desired page in
* 3. update the page table
* -----------------------------------------
* VPN: the VPN of the current logical address
* vm: pointer to the virtual memory space
* return: the physical mem page index of the desired page
*/
__device__ int ReplacePage(VirtualMemory* vm, int VPN) {
    *(vm->pagefault_num_ptr) += 1;
    int LRU_physicalIndex = FindLRUPageIndex(vm); // page index of the LRU page in physical mem
    int LRU_diskIndex = vm->invert_page_table[LRU_physicalIndex + vm->PAGE_ENTRIES]; // page index of the LRU page in disk (VPN)

    // swap LRU page out when the dirty bit is dirty, if dirty bit clean, no need to swap back to disk
    if (vm->invert_page_table[LRU_physicalIndex] & 0x1) {
        for (int j = 0; j < 32; j++) {
            (vm->storage + PageIndex2ByteOffset(LRU_diskIndex))[j] = (vm->buffer + PageIndex2ByteOffset(LRU_physicalIndex))[j];
        }
    }

    // swap the desired page in and set the dirty bit clean
    for (int j = 0; j < 32; j++) {
        (vm->buffer + PageIndex2ByteOffset(LRU_physicalIndex))[j] = (vm->storage + PageIndex2ByteOffset(VPN))[j];
        // set dirty bit: clean
        vm->invert_page_table[LRU_physicalIndex] &= 0xFFFFFFFE;
    }

    // update the page table
    vm->invert_page_table[LRU_physicalIndex] &= 0x7FFFFFFF; // set valid
    vm->invert_page_table[LRU_physicalIndex + vm->PAGE_ENTRIES] = VPN; // store the VPN in the page entry

    // return the page entry index of the VPN
    return LRU_physicalIndex;
}

/*
* find the LRU page, find the page linking with the maximum counter value
*/
__device__ int FindLRUPageIndex(VirtualMemory* vm) {
    u32 max;
    max = vm->pageTableCounter[0];
    int index = 0;
    for (int i = 0; i < 1024; i++) {
        if (vm->pageTableCounter[i] > max) {
            max = vm->pageTableCounter[i];
            index = i;
        }
    }
    return index;
}
