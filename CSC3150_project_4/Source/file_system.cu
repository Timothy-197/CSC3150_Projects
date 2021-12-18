#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ u32 FCB_FindFileIndex(FileSystem* fs, char* fileName);
__device__ bool FCB_CheckFileNameMatch(FileSystem* fs, char* fileName, int index);
__device__ u32 FCB_FindFileIndex_fpointer(FileSystem* fs, u32 fpointer);
__device__ bool FCB_CheckFpointerMatch(FileSystem* fs, u32 fpointer, u32 index);
__device__ u32 STORAGE_FindFreeSpaceIndex(FileSystem* fs);
__device__ void FCB_UpdateFCBEntry_Name(FileSystem* fs, u32 FCB_index, char* filename);
__device__ void FCB_UpdateFCBEntry_Fpointer(FileSystem* fs, u32 FCB_index, u32 fpointer);
__device__ void FCB_UpdateFCBEntry_Time(FileSystem* fs, u32 FCB_index, u32 time);
__device__ void FCB_UpdateFCBEntry_CreateTime(FileSystem* fs, u32 FCB_index, u32 createTime);
__device__ void FCB_UpdateFCBEntry_Size(FileSystem* fs, u32 FCB_index, u32 size);
__device__ void FCB_UpdateFCBEntry_DirtyBit(FileSystem* fs, u32 FCB_index, uchar dirtyBit);
__device__ void FCB_UpdateFCBEntry(FileSystem* fs, u32 FCB_index, char* filename, u32 fpointer, u32 time, u32 createTime, u32 size, uchar dirtyBit);
__device__ char* FCB_FetchEntry_Filename(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Fpointer(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Time(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_CreateTime(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Size(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_DirtyBit(FileSystem* fs, u32 FCB_index);
__device__ void Compaction(FileSystem* fs, u32 FCB_index);
__device__ void SUPERBLOCK_SetBit(FileSystem* fs, u32 fpointer, uchar bit);


/* File operation functions */
/*-----------------------------------------------------------------------------------*/

/*
* Initialize the memory structure
* sequence of the memories blocks (memory form low to high):
* superblock -> FCB block -> storage block
*/
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, u32 *TIME)
{
  // init variables
  fs->volume = volume; // disk memory for the project

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  fs->TIME = TIME;
}


/*
* Open a file: fin the fcb based on the name of the file
* the function will go through the superblock and try to find the FCB of the file
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk / s: the file name / op: access mode (G_WRITE --1 or G_READ --0)
* return value: the FCB number (index) of the corresponding file
*/
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	*(fs->TIME) += 1; // update time every operation
	if (op) { // write mode
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		if (FCB_index == fs->FCB_ENTRIES) { // no match file, need to create new FCB entry for non-existing file
			// find a free FCB block and initialize
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (FCB_FetchEntry_DirtyBit(fs, i)==0) { // check FCB entry dirty bit
					// find a clean FCB entry to set the FCB entry of the file
					// 1:new FCB entry: FCB index is i
					FCB_index = i;
				    // 2:find free space in the storage
					u32 startIndex = STORAGE_FindFreeSpaceIndex(fs);
					// 3:set the new FCB
					FCB_UpdateFCBEntry(fs, FCB_index, s, startIndex, 0, *(fs->TIME), 0, 1);
					// 4:since the file size now is 0, we don't need to set the bit map
					return FCB_index;
				}
			}
		}
		else { // there is a matched file, we need to do the compaction + allocate the new file
			Compaction(fs, FCB_index); // FCB index of this entry should not be changed
			u32 startIndex = STORAGE_FindFreeSpaceIndex(fs);
			// update FCB entry of the file
			FCB_UpdateFCBEntry(fs, FCB_index, s, startIndex, 0, FCB_FetchEntry_CreateTime(fs, FCB_index), 0, 1);
			return FCB_index;
		}
	}
	else { // read mode
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		if (FCB_index == fs->FCB_ENTRIES) { // cannot find read file, generate an error
			printf("cannot find the file to be read!");
			// need to raise an error
		}
		return FCB_index;
	}
}


/*
* Write file content to the disk
* the function will: 
* 1.find the file location based on the fp
* 2.free the file space on the disk, perform compaction (updating FCB and super block)
* 3.find a space on the disk (usually append to tail) and initialize file space (update FCB and super block)
* 4.write the file content to the disk
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk space / output: ouput buffer to store the read content
* / size: the size of bytes to be read from the disk / fp: FCB number of the file
*/
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	*(fs->TIME) += 1;
	// find the fpointer according to fp (FCB index)
	u32 fpointer = FCB_FetchEntry_Fpointer(fs, fp);
	u32 StoragePtr = fs->FILE_BASE_ADDRESS + fpointer * fs->STORAGE_BLOCK_SIZE;
	for (u32 k = 0; k < size; k++) {
		output[k] = *(fs->volume + StoragePtr + k);
	}
}

/*
* Read the file content
* the function will:
* 1.find the file location based on the fp
* 2.read the disk content into the output buffer
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk space / intput: ouput buffer to store the write content
* / size: the size of bytes to be written to the disk / fp: FCB number of the file
*/
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	*(fs->TIME) += 1;
	// 1. update FCB entry: size, time (name, fpointer and dirty bit has been set)
	FCB_UpdateFCBEntry_Size(fs, fp, size);
	FCB_UpdateFCBEntry_Time(fs, fp, *(fs->TIME));
	
	// 2. write the content into the storage
	u32 fpointer = FCB_FetchEntry_Fpointer(fs, fp);
	for (int i = 0; i < size; i++) {
		*(fs->volume + fs->FILE_BASE_ADDRESS + fpointer * fs->STORAGE_BLOCK_SIZE + i) = input[i];
	}

	// 3. update the bit map
	int blockCount;
	if (size % fs->STORAGE_BLOCK_SIZE > 0) blockCount = size / fs->STORAGE_BLOCK_SIZE + 1;
	else blockCount = size / fs->STORAGE_BLOCK_SIZE;
	for (u32 i = 0; i < blockCount; i++) {
		SUPERBLOCK_SetBit(fs, fpointer+i, 1);
	}
}


/*
* LS command
* the function will list the files, basing on the modified date (LS_D), file size (LS_S)
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk space / op: operation code: LS_D --0 or LS_S --1
*/
__device__ void fs_gsys(FileSystem *fs, int op)
{
	*(fs->TIME) += 1;
	u32 DityIndexes[1024]; // temp array to record the indexes of the dirty FCB entries
	int count = 0;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i)) {//dirty entry
			DityIndexes[count] = i;
			count++;
		}
	}

	// sort: use bubble sort algorithm
	if (op == 0) {// date
		// sort by time, the first FCB index after sort is the lru file
		for (int i = 0; i < count-1; i++) {
			for (int j = 0; j < count - 1 - i; j++) {
				if (FCB_FetchEntry_Time(fs, DityIndexes[j]) < FCB_FetchEntry_Time(fs, DityIndexes[j + 1])) {
					// swap the indexes in the dirtyindexes array
					u32 temp = DityIndexes[j];
					DityIndexes[j] = DityIndexes[j + 1];
					DityIndexes[j + 1] = temp;
				}
			}
		}
		// print out
		printf("=== Sort by modified time ===\n");
		for (int i = 0; i < count; i++) {
			printf("%s\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]));
		}
	}
	else if (op == 1) {// size
		// sort by size, the first FCB index after sort is the  file
		for (int i = 0; i < count - 1; i++) {
			for (int j = 0; j < count - 1 - i; j++) {
				if (FCB_FetchEntry_Size(fs, DityIndexes[j]) < FCB_FetchEntry_Size(fs, DityIndexes[j + 1])) {
					// swap the indexes in the dirtyindexes array
					u32 temp = DityIndexes[j];
					DityIndexes[j] = DityIndexes[j + 1];
					DityIndexes[j + 1] = temp;
				}
				else if (FCB_FetchEntry_Size(fs, DityIndexes[j]) == FCB_FetchEntry_Size(fs, DityIndexes[j + 1])) {
					// same size, the first created one comes (smaller time value) first
					if (FCB_FetchEntry_CreateTime(fs, DityIndexes[j]) > FCB_FetchEntry_CreateTime(fs, DityIndexes[j + 1])) {
						u32 temp = DityIndexes[j];
						DityIndexes[j] = DityIndexes[j + 1];
						DityIndexes[j + 1] = temp;
					}
				}
			}
		}
		// print out
		printf("=== Sort by file size ===\n");
		for (int i = 0; i < count; i++) {
			printf("%s %d\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]), FCB_FetchEntry_Size(fs, DityIndexes[i]));
		}
	}
	else {
		printf("Wrong input code!");
	}
}

/*
* RM command
* the function will find the FCB of the file by the file name
* the function will free the file space of the corresponding file in the disk
* after freeing the space, the function will do compaction (update FCB and superblock)
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk space / op: operation code: RM --2
*/
__device__ void fs_gsys(FileSystem* fs, int op, char* s)
{
	*(fs->TIME) += 1;
	if (op == 2) {
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		if (FCB_index == fs->FCB_ENTRIES) {
			printf("no file to be removed!");
			return;
		}
		// do the compaction to remove the file
		Compaction(fs, FCB_index);
	}
	else {
		printf("Invalid command!\n");
		return;
	}
}


/* Functions to facilate implementation */
/*-----------------------------------------------------------------------------------*/

/*
* Find the FCB index by the name of the file
* ----------------------------------------------------
* return value: the FCB index of the file
*/
__device__ u32 FCB_FindFileIndex(FileSystem *fs, char *fileName)
{
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check dirty bit of FCB entry
			continue; // no need to check name match, since the FCB entry is not being used
		}
		else if (FCB_CheckFileNameMatch(fs, fileName, i)) { // if dirty, check name match
			// return the FCB index of the FCB entry of the file
			return i;
		}
	}
	return fs->FCB_ENTRIES; // return 1024 to indicate that no file match can be found
}
//Check if the name in the FCB blocks matches the filename
__device__ bool FCB_CheckFileNameMatch(FileSystem* fs, char* fileName, int index)
{
	for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		if (*(fs->volume + fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE + i) != fileName[i]) {
			// index: block index in the FCB block, i: the byte index in the FCB entry
			return false; // when the character mismatch return false
		}
		if (fileName[i] == '\0') {
			return true; // check if this is the end of the filename string
		}
	}
	return false; // the filename array must have the '\0' to indicate the end
}


/*
* This function will search for the FCB index of according to the fpointer (start block index)
* The function is similar to the FCB_FindFileIndex(FileSystem *fs, char *fileName)
* ----------------------------------------------------------------------------------
* return value: the FCB index of the file
*/
__device__ u32 FCB_FindFileIndex_fpointer(FileSystem* fs, u32 fpointer)
{
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check dirty bit of FCB entry
			continue; // no need to check name match, since the FCB entry is not being used
		}
		else if (FCB_CheckFpointerMatch(fs, fpointer, i)) {
			return i;
		}
	}
	return fs->FCB_ENTRIES; // return 1024 to indicate that no file match can be found
}

// check of there is a fpointer match in the fcb block
__device__ bool FCB_CheckFpointerMatch(FileSystem* fs, u32 fpointer, u32 index)
{
	u32 temp = FCB_FetchEntry_Fpointer(fs, index);
	return (temp == fpointer);
}

/*
* The function will find the starting index in the storage block of the free space
* Since I compact all the storage content, the space after this index is always free
* The function go through the superblock to find the index
* ----------------------------------------------------
* return value: the starting index of the free space (unit of the index: block in the storage)
* this index also maps to the bit of the bit map (super block)
*/
__device__ u32 STORAGE_FindFreeSpaceIndex(FileSystem* fs)
{
	// traverse the superblock and find the first 0 bit
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		if (*(fs->volume + i) == 0xFF) continue; // when the bit map are filled
		else { // find the starting index
			for (int k = 0; k < 8; k++) {
				if (!((*(fs->volume + i) >> k) & 0x1)) {
					// find the zero bit, return the index
					return (i * 8 + k);
				}
			}
		}
	}
	return fs->SUPERBLOCK_SIZE * 8; // starting index overflow
}


/*
* The following group of functions are to update the FCB entry with given parameters:
* fs: mem pointer / FCB_index: the FCB index of the file 
* / filename: file name [19:0] / fpointer: file pointer [23:20] / time: access time [27:24] /
* size [29:28] / dirtyBit: dirty bit of FCB entry: [31] 
*/
__device__ void FCB_UpdateFCBEntry_Name(FileSystem *fs, u32 FCB_index, char *filename)
{
	// set filename [19:0]
	for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + i) = filename[i];
		if (filename[i] == '\0') break; // finish setting name
	}
}
__device__ void FCB_UpdateFCBEntry_Fpointer(FileSystem* fs, u32 FCB_index, u32 fpointer)
{
	// set fpointer --byte[21:20], note that bit[7] of byte[21] is the dirty bit
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 20 + 0) = (uchar)(fpointer & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 20 + 1) &= 0x80; // clear the last 7 bits
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 20 + 1) |= (uchar)((fpointer >> BYTE_OFFSET) & 0x7F);
}
__device__ void FCB_UpdateFCBEntry_Time(FileSystem* fs, u32 FCB_index, u32 time)
{
	// set access time --byte[25:22]
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 22 + 0) = (uchar)(time & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 22 + 1) = (uchar)((time >> BYTE_OFFSET) & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 22 + 2) = (uchar)((time >> (BYTE_OFFSET * 2)) & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 22 + 3) = (uchar)((time >> (BYTE_OFFSET * 3)) & 0xFF);
}
__device__ void FCB_UpdateFCBEntry_CreateTime(FileSystem* fs, u32 FCB_index, u32 createTime)
{
	// set create time --byte[29:26]
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 26 + 0) = (uchar)(createTime & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 26 + 1) = (uchar)((createTime >> BYTE_OFFSET) & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 26 + 2) = (uchar)((createTime >> (BYTE_OFFSET * 2)) & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 26 + 3) = (uchar)((createTime >> (BYTE_OFFSET * 3)) & 0xFF);
}
__device__ void FCB_UpdateFCBEntry_Size(FileSystem* fs, u32 FCB_index, u32 size)
{
	// set size --byte[31:30]
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 30 + 0) = (uchar)(size & 0xFF);
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 30 + 1) = (uchar)((size >> BYTE_OFFSET) & 0xFF);
}
__device__ void FCB_UpdateFCBEntry_DirtyBit(FileSystem* fs, u32 FCB_index, uchar dirtyBit)
{
	// set dirty bit --bit[7] of byte[21]
	if (dirtyBit == 0) *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 21) &= 0x7F;
	else if (dirtyBit == 1) *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 21) |= 0x80;
}
__device__ void FCB_UpdateFCBEntry(FileSystem* fs, u32 FCB_index, char* filename, u32 fpointer, u32 time, u32 createTime, u32 size, uchar dirtyBit) 
{
	FCB_UpdateFCBEntry_Name(fs, FCB_index, filename);
	FCB_UpdateFCBEntry_Fpointer(fs, FCB_index, fpointer);
	FCB_UpdateFCBEntry_Time(fs, FCB_index, time);
	FCB_UpdateFCBEntry_CreateTime(fs, FCB_index, createTime);
	FCB_UpdateFCBEntry_Size(fs, FCB_index, size);
	FCB_UpdateFCBEntry_DirtyBit(fs, FCB_index, dirtyBit);
}

/*
* The following group of functions are to retrieve the value from the FCB entry
* parameter: the FCB index (fp)
* return value: depends on the case
*/
__device__ char* FCB_FetchEntry_Filename(FileSystem *fs, u32 FCB_index) 
{
	char temp[20];
	int count = 0;
	for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
		count++;
		temp[i] = *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + i);
		if (*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + i) == '\0') break; // finish setting name
	}
	return temp;
}
__device__ u32 FCB_FetchEntry_Fpointer(FileSystem* fs, u32 FCB_index)
{
	//[21:20], exculde the bit[7] of byte[21]
	u32 temp = 0;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 20 + 0);
	temp |= (*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 20 + 1) & 0x7F) << BYTE_OFFSET;
	return temp;
}
__device__ u32 FCB_FetchEntry_Time(FileSystem *fs, u32 FCB_index)
{
	//[25:22]
	u32 temp = 0;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 22 + 0);
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 22 + 1) << BYTE_OFFSET;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 22 + 2) << (BYTE_OFFSET * 2);
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 22 + 3) << (BYTE_OFFSET * 3);
	return temp;
}
__device__ u32 FCB_FetchEntry_CreateTime(FileSystem* fs, u32 FCB_index)
{
	//[29:26]
	u32 temp = 0;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 26 + 0);
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 26 + 1) << BYTE_OFFSET;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 26 + 2) << (BYTE_OFFSET * 2);
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * (fs->FCB_SIZE) + 26 + 3) << (BYTE_OFFSET * 3);
	return temp;
}
__device__ u32 FCB_FetchEntry_Size(FileSystem* fs, u32 FCB_index)
{
	// [31:30]
	u32 temp = 0;
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 30 + 0);
	temp |= *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 30 + 1) << BYTE_OFFSET;
	return temp;
}
__device__ uchar FCB_FetchEntry_DirtyBit(FileSystem *fs, u32 FCB_index)
{
	// byte[21], bit 7
	return (*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 21) >> 7);
}

/*
* Free the space of file content and do the compaction
* the function will be called during writing process and removing process
* the function will
* 1. clear the file content in storage, set the dirty bit to clean of the FCB entry of the removed file
* 2. remove the blank space, compact file content in storage, update FCB entries (fpointer)
* -----------------------------------------------------------------------------------------
* parameters: fs: filesystem pointer / FCB_index: fp of the file to be removed
*/
__device__ void Compaction(FileSystem* fs, u32 FCB_index)
{
	u32 moveOffset; // the block offset for the memories to move in the compaction
	u32 moveByteOffset = FCB_FetchEntry_Size(fs, FCB_index); // file size unit: byte
	if (moveByteOffset % 32) moveOffset = moveByteOffset>>5 + 1; // has internal fragmentation
	else moveOffset = moveByteOffset>>5;

	u32 startBlockIndex = FCB_FetchEntry_Fpointer(fs, FCB_index); // block index of the removed file
	u32 startAddr = startBlockIndex * (fs->STORAGE_BLOCK_SIZE) + fs->FILE_BASE_ADDRESS; // start address to relocate
	u32 endBlockIndex = STORAGE_FindFreeSpaceIndex(fs);
	// 1.update the FCB entry of the removed file
	FCB_UpdateFCBEntry_DirtyBit(fs, FCB_index, 0); // set dirty bit clean, free the FCB entry

	// 2.update the storage content, and update the fpointers of the affected FCB entries
	for (u32 i = startBlockIndex; i < endBlockIndex-moveOffset; i++) {
		// copy the block[i+offset] to the block[i] (byte by byte)
		for (int j = 0; j < 32; j++) {
			*(fs->volume + startAddr + i * fs->STORAGE_BLOCK_SIZE + j) = *(fs->volume + startAddr + (i + moveOffset) * fs->STORAGE_BLOCK_SIZE + j);
		}
		// update the FCB entry of each affected file
		int currentFCBIndex = FCB_FindFileIndex_fpointer(fs, i+moveOffset);
		if (currentFCBIndex != fs->FCB_ENTRIES) {// finds the corresponding file entry
			// update the fpointer in the FCB entry
			FCB_UpdateFCBEntry_Fpointer(fs, currentFCBIndex, i); // fpointer in the entry will decrease
		}
	}
	
	// 3. update the superblock (bit map), only need to clear the last few bits
	for (int i = startBlockIndex; i < endBlockIndex-moveOffset; i++) {
		SUPERBLOCK_SetBit(fs, i, 1);
	}
	for (int i = 0; i < moveOffset; i++) {
		SUPERBLOCK_SetBit(fs, endBlockIndex-i-1, 0);
	}
}


/*
* Set / Reset the bit of bitmap in the superblock
* -----------------------------------------------------
* parameter: fpointer: the fpointer (block index) of the file
* bit: 1: set (occupied), 0: reset (free)
*/
__device__ void SUPERBLOCK_SetBit(FileSystem *fs, u32 fpointer, uchar bit)
{
	u32 byteIndex = fpointer / BYTE_OFFSET;
	u32 bitIndex = fpointer % BYTE_OFFSET;
	if (bit) *(fs->volume + byteIndex) |= (0x1 << bitIndex); // set bit
	else { // reset bit
		uchar temp = ~(0x1 << bitIndex);
		*(fs->volume + byteIndex) &= temp;
	}
}

