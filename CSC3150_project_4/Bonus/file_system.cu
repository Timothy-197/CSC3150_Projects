#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0; // I do not use this
__device__ __managed__ uchar dir1Index = 0; // a number to record how many directories have been created
__device__ __managed__ uchar dir2Index = 0;
__device__ __managed__ uchar dir3Index = 0;


__device__ u32 FCB_FindFileIndex(FileSystem* fs, char* fileName);
__device__ u32 FCB_FindFileIndex_fpointer(FileSystem* fs, u32 fpointer);
__device__ u32 FCB_FindDirectoryIndex_Name(FileSystem* fs, char* directoryName);
__device__ u32 FCB_FindDirectoryIndex_DirInfo(FileSystem* fs, u32 dirInfo, uchar dirMap);
__device__ bool FCB_CheckFpointerMatch(FileSystem* fs, u32 fpointer, u32 index);
__device__ bool FCB_CheckFileNameMatch(FileSystem* fs, char* fileName, int index);
__device__ bool FCB_CheckDirectoryMatch_Current(FileSystem* fs, int index);
__device__ bool FCB_CheckDirectoryMatch_Compare(FileSystem* fs, int index, u32 directoryInfo);

__device__ u32 STORAGE_FindFreeSpaceIndex(FileSystem* fs);

__device__ void FCB_UpdateFCBEntry_Name(FileSystem* fs, u32 FCB_index, char* filename);
__device__ void FCB_UpdateFCBEntry_Fpointer(FileSystem* fs, u32 FCB_index, u32 fpointer);
__device__ void FCB_UpdateFCBEntry_Time(FileSystem* fs, u32 FCB_index, u32 time);
__device__ void FCB_UpdateFCBEntry_CreateTime(FileSystem* fs, u32 FCB_index, u32 createTime);
__device__ void FCB_UpdateFCBEntry_Size(FileSystem* fs, u32 FCB_index, u32 size);
__device__ void FCB_UpdateFCBEntry_DirtyBit(FileSystem* fs, u32 FCB_index, uchar dirtyBit);
__device__ void FCB_UpdateFCBEntry_DirectoryBit(FileSystem* fs, u32 FCB_index, uchar directoryBit);
__device__ void FCB_UpdateFCBEntry_Directory1(FileSystem* fs, u32 FCB_index, uchar directoryIndex0);
__device__ void FCB_UpdateFCBEntry_Directory2(FileSystem* fs, u32 FCB_index, uchar directoryIndex1);
__device__ void FCB_UpdateFCBEntry_Directory3(FileSystem* fs, u32 FCB_index, uchar directoryIndex2);
__device__ void FCB_UpdateFCBEntry_DirectoryMap(FileSystem* fs, u32 FCB_index, uchar directoryIndex);
__device__ void FCB_UpdateFCBEntry_Depth(FileSystem* fs, u32 FCB_index, uchar depth);
__device__ void FCB_UpdateFCBEntry(FileSystem* fs, u32 FCB_index, char* filename,
	u32 fpointer, u32 time, u32 createTime, u32 size, uchar dirtyBit,
	uchar directoryBit, uchar directoryIndex1, uchar directoryIndex2, uchar directoryIndex3, uchar directoryIndex, uchar depth);

__device__ char* FCB_FetchEntry_Filename(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Fpointer(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Time(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_CreateTime(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Size(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_DirtyBit(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_DirectoryBit(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_Directory1(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_Directory2(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_Directory3(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_DirectoryMap(FileSystem* fs, u32 FCB_index);
__device__ u32 FCB_FetchEntry_Dir_All(FileSystem* fs, u32 FCB_index);
__device__ uchar FCB_FetchEntry_Depth(FileSystem* fs, u32 FCB_index);

__device__ void Compaction(FileSystem* fs, u32 FCB_index);
__device__ void SUPERBLOCK_SetBit(FileSystem* fs, u32 fpointer, uchar bit);
__device__ void RmDirectory(FileSystem* fs, u32 FCB_index);

__device__ void DIR_SetDirectoryInfo(FileSystem* fs, uchar depth, uchar directory1, uchar directory2, uchar directory3);
__device__ uchar DIR_FetchDirectory(FileSystem* fs, int depth);
__device__ u32 DIR_FetchDirectory_All(FileSystem* fs);
__device__ uchar DIR_FetchDepth(FileSystem* fs);
__device__ u32 DIR_GetParentDirInfo(FileSystem* fs);
__device__ void DIR_Move2Parent(FileSystem* fs);
__device__ void DIR_Move2Destination(FileSystem* fs, u32 FCB_index);
__device__ u32 DIR_FileGetParent(FileSystem* fs, u32 FCB_index);
__device__ u32 DIR_FileGetDir(FileSystem* fs, u32 FCB_index);

__device__ int STR_Length(char* s);


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
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, u32 *TIME, u32 *DIRECTORY)
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
  fs->DIRECTORY = DIRECTORY;
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
	*(fs->TIME) += 1;
	if (op) { // write mode
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		//printf("openfile: the FCB index found is: %d\n", FCB_index);
		if (FCB_index == fs->FCB_ENTRIES) { // no match file, need to create new FCB entry for non-existing file
			// find a free FCB block and initialize
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check FCB entry dirty bit
					// find a clean FCB entry to set the FCB entry of the file
					// 1:new FCB entry: FCB index is i
					FCB_index = i;
				    // 2:find free space in the storage
					u32 startIndex = STORAGE_FindFreeSpaceIndex(fs);
					//printf("openfile: try to allocate new file, the new startIndex is : %d\n", startIndex);
					// 3:set the new FCB
					FCB_UpdateFCBEntry(fs, FCB_index, s, startIndex, 0, *(fs->TIME), 0, 1, 0,
						DIR_FetchDirectory(fs, 1), DIR_FetchDirectory(fs, 2), DIR_FetchDirectory(fs, 3),
						0, DIR_FetchDepth(fs));
					// 4:since the file size now is 0, we don't need to set the bit map
					// 5.we should update the size of the corresponding directory
					int strLen = STR_Length(s);
					FCB_UpdateFCBEntry_Size(fs, DIR_FileGetDir(fs, FCB_index), strLen);
					return FCB_index;
				}
			}
		}
		else { // there is a matched file, we need to do the compaction + allocate the new file
			Compaction(fs, FCB_index); // FCB index of this entry should not be changed
			u32 startIndex = STORAGE_FindFreeSpaceIndex(fs);
			// update FCB entry of the file
			FCB_UpdateFCBEntry(fs, FCB_index, s, startIndex, 0, FCB_FetchEntry_CreateTime(fs, FCB_index), 0, 1, 0,
				DIR_FetchDirectory(fs, 1), DIR_FetchDirectory(fs, 2), DIR_FetchDirectory(fs, 3),
				0, DIR_FetchDepth(fs));
			return FCB_index;
		}
	}
	else { // read mode
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		if (FCB_index == fs->FCB_ENTRIES) { // cannot find read file, generate an error
			printf("Openfile: cannot find the file to be read!");
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
	//printf("Write: before write, the bit map is %d\n", *(fs->volume));
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
__device__ void fs_gsys(FileSystem* fs, int op)
{
	// sort: use bubble sort algorithm
	if (op == 0) {// date
		u32 DityIndexes[1024]; // temp array to record the indexes of the dirty FCB entries
		int count = 0;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			if (FCB_FetchEntry_DirtyBit(fs, i) && FCB_CheckDirectoryMatch_Current(fs, i)) {//dirty entry & match the current directory
				DityIndexes[count] = i;
				count++;
			}
		}
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
			if (FCB_FetchEntry_DirectoryBit(fs, DityIndexes[i])) printf("%s d\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]));
			else printf("%s\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]));
		}
		return;
	}
	else if (op == 1) {// size
		u32 DityIndexes[1024]; // temp array to record the indexes of the dirty FCB entries
		int count = 0;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			if (FCB_FetchEntry_DirtyBit(fs, i) && FCB_CheckDirectoryMatch_Current(fs, i)) {//dirty entry
				DityIndexes[count] = i;
				count++;
			}
		}
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
			if (FCB_FetchEntry_DirectoryBit(fs, DityIndexes[i])) printf("%s %d d\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]), FCB_FetchEntry_Size(fs, DityIndexes[i]));
			else printf("%s %d\n", FCB_FetchEntry_Filename(fs, DityIndexes[i]), FCB_FetchEntry_Size(fs, DityIndexes[i]));
		}
		return;
	}
	else if (op == 3) {// CD_P
		*(fs->TIME) += 1;
		DIR_Move2Parent(fs);
		// update directory time
		if (DIR_FetchDirectory_All(fs) != 0) { // if not root directory
			u32 parentDirInfo = DIR_GetParentDirInfo(fs);
			uchar parentDirMap = DIR_FetchDirectory(fs, DIR_FetchDepth(fs));
			u32 dirIndex = FCB_FindDirectoryIndex_DirInfo(fs, parentDirInfo, parentDirMap);
			if (dirIndex != fs->FCB_ENTRIES) { // other wise it should be root
				FCB_UpdateFCBEntry_Time(fs, dirIndex, *(fs->TIME));
			}
		}
		return;
	}
	else if (op == 4) { // PWD
		// 1. find the directory entry which points to the current directory
		u32 tempDirPtrValue = *(fs->DIRECTORY); // store the value of the current dirInfo

		uchar currDepth = DIR_FetchDepth(fs);
		if (currDepth == 0) {
			// root
			printf("/\n");
			return;
		}
		else {
			char sStack[60]; // temp string to hold a stack of strings
			uchar iter = currDepth;
			uchar currDirMap;
			while (iter > 0) {
				// find the fcb index according to the depth and dirMap to the pointing directory
				// get dirMap of the directory
				currDirMap = DIR_FetchDirectory(fs, iter);
				// move the directory pointer to the parent
				DIR_Move2Parent(fs);
				// get the FCB index of the directory entry pointing to current directory
				u32 FCB_index = FCB_FindDirectoryIndex_DirInfo(fs, DIR_FetchDirectory_All(fs), currDirMap);
				if (FCB_index == fs->FCB_ENTRIES) {
					printf("Invalid path!\n");
					return;
				}
				// store the pwd string in temp buffer
				for (int i = 0; i < 20; i++) {
					char currC = FCB_FetchEntry_Filename(fs, FCB_index)[i];
					sStack[20 * (iter - 1) + i] = currC;
					if (currC == '\0') {
						break;
					}
				}
				// update the current directory: depth and dirmap
				iter--;
			}
			// print out the directory information
			for (int i = 0; i < currDepth; i++) {
				printf("/%s",sStack+20*i);
			}
			printf("\n");
			// restore the directory pointer
			*(fs->DIRECTORY) = tempDirPtrValue;
		}
		return;
	}
	else {
		printf("Wrong input code! code is %d\n", op);
		return;
	}
}

/*
* Commands: RM, MKDIR
* RM: remove a file based on the input file name
* MKDIR: create a directory and enter that direcotry
* ----------------------------------------------------------------------------------
* parameters: fs: pointer to the disk space / op: operation code: RM --2, MKDIR --5
*/
__device__ void fs_gsys(FileSystem* fs, int op, char* s)
{
	if (op == 2) { // RM
		u32 FCB_index = FCB_FindFileIndex(fs, s);
		if (FCB_index == fs->FCB_ENTRIES) {
			printf("no file to be removed!");
			return;
		}
		// do the compaction to remove the file
		Compaction(fs, FCB_index);
		// update the size of the corresponding directory entry (since the file name is removed)
		int strLen = STR_Length(s);
		FCB_UpdateFCBEntry_Size(fs, DIR_FileGetDir(fs, FCB_index), 
			FCB_FetchEntry_Size(fs, DIR_FileGetDir(fs, FCB_index)) - strLen);
		return;
	}
	else if (op == 5) { // MKDIR
		*(fs->TIME) += 1; // update time
		// 1. Create directory: update the FCB entry of the directory
		// do not change storage and superblock

		// 2. create new FCB entry for the directory
		uchar currDepth = DIR_FetchDepth(fs);
		if (currDepth >= 3) {
			printf("cannot create directory for the depth limit");
			return;
		}
		// update dirtectory indexes & fetch the new dirtory index for the directory
		uchar dir1, dir2, dir3; // dirX is the direcotry that this entry is pointing to the current directory
		uchar dirMap;
		if (currDepth == 0) { 
			dir1Index += 1; 
			// update the directory index, 
			// the other parts of the dir info of the entry should be the same as current dirctory
			dir1 = dir1Index;
			dir2 = 0;
			dir3 = 0;
			dirMap = dir1;
		}
		else if (currDepth == 1) {
			dir2Index += 1;
			dir1 = DIR_FetchDirectory(fs, 1);
			dir2 = dir2Index;
			dir3 = 0;
			dirMap = dir2;
		}
		else if (currDepth == 2) {
			dir3Index += 1;
			dir1 = DIR_FetchDirectory(fs, 1);
			dir2 = DIR_FetchDirectory(fs, 2);
			dir3 = dir3Index;
			dirMap = dir3;
		}
		// create the FCB entry for the directory
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			if (FCB_FetchEntry_DirtyBit(fs, i) == 0) {
				// find a clean FCB entry to update
				u32 FCB_index = i;
				FCB_UpdateFCBEntry(fs, FCB_index, s, 0, *(fs->TIME), *(fs->TIME), 0, 1, 1,
					DIR_FetchDirectory(fs, 1), DIR_FetchDirectory(fs, 2), DIR_FetchDirectory(fs, 3), dirMap, currDepth);
				break;
			}
		}
		return;
	}
	else if (op == 6) {// CD
		// 1. find the FCB entry of the directory
		*(fs->TIME) += 1;
		u32 FCB_index = FCB_FindDirectoryIndex_Name(fs, s); // find FCB index of the directory
		if (FCB_index == fs->FCB_ENTRIES) {
			printf("cannot find the directory!\n");
			return;
		}
		// 2. enter the directory: update the fs->DIRECTORY, depth++, get dirmap from the entry
		DIR_Move2Destination(fs, FCB_index);
		return;
	}
	else if (op == 7) { // RM_RF
		u32 tempDir = DIR_FetchDirectory_All(fs);
		u32 dirIndex = FCB_FindDirectoryIndex_Name(fs, s);
		if (dirIndex == 1024) { // not find the drectory
			printf("no such directory!\n");
			return;
		}
		RmDirectory(fs, dirIndex);
		// resote the directory pointer
		*(fs->DIRECTORY) = tempDir;
		return;
	}
	else {
		printf("Invalid command!\n");
		return;
	}
}


/* Functions to faciliate implementation */
/*-----------------------------------------------------------------------------------*/

/*
* Find the directory entry fcb index according to the name of the directory
* the name + the directory information of the directory -> unique directory
* ------------------------------------------------------------------------------
* return: fcb index of the directory
*/
__device__ u32 FCB_FindDirectoryIndex_Name(FileSystem* fs, char* directoryName)
{
	for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check dirty bit of FCB entry
			continue; // no need to check name match, since the FCB entry is not being used
		}
		else if (FCB_CheckFileNameMatch(fs, directoryName, i)) { // if dirty, check name match
			if (FCB_CheckDirectoryMatch_Current(fs, i) && (FCB_FetchEntry_DirectoryBit(fs, i))) {
				// check if the directory structure matches & check if the FCB corresponds to directory type
				return i; // return if the entry matches, else continue
			}
		}
	}
	return fs->FCB_ENTRIES; // return 1024 to indicate that no file match can be found
}
/*
* Find the directory entry fcb index according to: 
* the directory information of the directory / dirMap of the directory (which is unique)
* ---------------------------------------------------------------------------------------------
* return: fcb index of the directory
*/
__device__ u32 FCB_FindDirectoryIndex_DirInfo(FileSystem* fs, u32 dirInfo, uchar dirMap) 
{
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check dirty bit of FCB entry
			continue; // no need to check name match, since the FCB entry is not being used
		}
		if ((FCB_FetchEntry_DirectoryBit(fs, i) == 1) 
			&& FCB_CheckDirectoryMatch_Compare(fs, i, dirInfo) 
			&& (FCB_FetchEntry_DirectoryMap(fs, i) == dirMap)) {
			// check if the entry is a directory entry + check the depth and the dirmap
			return i;
		}
	}
	return fs->FCB_ENTRIES; // return 1024 to indicate that no file match can be found
}

/*
* Find the FCB index by the name of the file
* ----------------------------------------------------
* return value: the FCB index of the file
*/
__device__ u32 FCB_FindFileIndex(FileSystem* fs, char* fileName)
{
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		if (FCB_FetchEntry_DirtyBit(fs, i) == 0) { // check dirty bit of FCB entry
			continue; // no need to check name match, since the FCB entry is not being used
		}
		else if (FCB_CheckFileNameMatch(fs, fileName, i)) { // if dirty, check name match
			// return the FCB index of the FCB entry of the file
			if (FCB_CheckDirectoryMatch_Current(fs, i) && (FCB_FetchEntry_DirectoryBit(fs, i) == 0)) {
				// check if the directory structure matches & check if the FCB corresponds to file type
				return i; // return if the entry matches, else continue
			}
		}
	}
	return fs->FCB_ENTRIES; // return 1024 to indicate that no file match can be found
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

/*
* the following functions implement the comparsion during the finding process
*/
// check of there is a fpointer match in the fcb block
__device__ bool FCB_CheckFpointerMatch(FileSystem* fs, u32 fpointer, u32 index)
{
	u32 temp = FCB_FetchEntry_Fpointer(fs, index);
	return (temp == fpointer);
}
// Check if the name in the FCB blocks matches the filename (or directory name)
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
// Check if the directory information in the FCB matches the current directory information
__device__ bool FCB_CheckDirectoryMatch_Current(FileSystem* fs, int index)
{
	u32 temp = FCB_FetchEntry_Dir_All(fs, index);
	return (*(fs->DIRECTORY) == temp);
}
// check if the directory dirMap matches the current FCB entry
__device__ bool FCB_CheckDirectoryMatch_Compare(FileSystem* fs, int index, u32 directoryInfo)
{
	u32 temp = 0; // temp variable for comparision
	temp |= FCB_FetchEntry_Depth(fs, index);
	temp |= FCB_FetchEntry_Directory1(fs, index) << BYTE_OFFSET;
	temp |= FCB_FetchEntry_Directory2(fs, index) << (BYTE_OFFSET * 2);
	temp |= FCB_FetchEntry_Directory3(fs, index) << (BYTE_OFFSET * 3);
	return (directoryInfo == temp);
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
__device__ void FCB_UpdateFCBEntry_Name(FileSystem* fs, u32 FCB_index, char* filename)
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
__device__ void FCB_UpdateFCBEntry_DirectoryBit(FileSystem* fs, u32 FCB_index, uchar directoryBit)
{
	// set directory bit --byte[37], 0 indicates file, 1 indicates directory
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 37) = directoryBit;
}
__device__ void FCB_UpdateFCBEntry_Directory1(FileSystem* fs, u32 FCB_index, uchar directoryIndex1)
{
	// set directory information with depth 1 --byte[32]
	// note that 0 indicates the root directory
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 32) = directoryIndex1;
}
__device__ void FCB_UpdateFCBEntry_Directory2(FileSystem* fs, u32 FCB_index, uchar directoryIndex2)
{
	// set directory information with depth 2 --byte[33]
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 33) = directoryIndex2;
}
__device__ void FCB_UpdateFCBEntry_Directory3(FileSystem* fs, u32 FCB_index, uchar directoryIndex3)
{
	// set directory information with depth 3 --byte[34]
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 34) = directoryIndex3;
}
__device__ void FCB_UpdateFCBEntry_DirectoryMap(FileSystem* fs, u32 FCB_index, uchar directoryIndex)
{
	// set directory map if the current FCB corresponds to a directory --byte[35]
	// if the FCB maps to a directory
	//	the directory index will map to the directory name stored in the FCB
	// if the FCB maps to a file
	//	set the directory index 0 by default
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 35) = directoryIndex;
}
__device__ void FCB_UpdateFCBEntry_Depth(FileSystem* fs, u32 FCB_index, uchar depth)
{
	// set the depth of the corresponding file or directory -- bit[36]
	// 0 indicates root, 1: depth1, 2: depth2, 3: depth3
	*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 36) = depth;
}
__device__ void FCB_UpdateFCBEntry(FileSystem* fs, u32 FCB_index, char* filename, 
	u32 fpointer, u32 time, u32 createTime, u32 size, uchar dirtyBit, 
	uchar directoryBit, uchar directoryIndex1, uchar directoryIndex2, uchar directoryIndex3, uchar directoryIndex, uchar depth) 
{
	FCB_UpdateFCBEntry_Name(fs, FCB_index, filename);
	FCB_UpdateFCBEntry_Fpointer(fs, FCB_index, fpointer);
	FCB_UpdateFCBEntry_Time(fs, FCB_index, time);
	FCB_UpdateFCBEntry_CreateTime(fs, FCB_index, createTime);
	FCB_UpdateFCBEntry_Size(fs, FCB_index, size);
	FCB_UpdateFCBEntry_DirtyBit(fs, FCB_index, dirtyBit);
	FCB_UpdateFCBEntry_DirectoryBit(fs, FCB_index, directoryBit);
	FCB_UpdateFCBEntry_Directory1(fs, FCB_index, directoryIndex1);
	FCB_UpdateFCBEntry_Directory2(fs, FCB_index, directoryIndex2);
	FCB_UpdateFCBEntry_Directory3(fs, FCB_index, directoryIndex3);
	FCB_UpdateFCBEntry_DirectoryMap(fs, FCB_index, directoryIndex);
	FCB_UpdateFCBEntry_Depth(fs, FCB_index, depth);
}
// update all the firectory information
__device__ void FCB_UpdateFCBEntry_Dir_All(FileSystem* fs, u32 FCB_index, u32 dirInfo)
{
	FCB_UpdateFCBEntry_Depth(fs, FCB_index, (uchar)(dirInfo & 0xFF));
	FCB_UpdateFCBEntry_Directory1(fs, FCB_index, (uchar)((dirInfo >> BYTE_OFFSET) & 0xFF));
	FCB_UpdateFCBEntry_Directory2(fs, FCB_index, (uchar)((dirInfo >> (BYTE_OFFSET*2)) & 0xFF));
	FCB_UpdateFCBEntry_Directory3(fs, FCB_index, (uchar)((dirInfo >> (BYTE_OFFSET*3)) & 0xFF));
}


/*
* The following group of functions are to retrieve the value from the FCB entry
* parameter: the FCB index (fp)
* return value: depends on the case
*/
__device__ char* FCB_FetchEntry_Filename(FileSystem* fs, u32 FCB_index)
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
__device__ u32 FCB_FetchEntry_Time(FileSystem* fs, u32 FCB_index)
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
__device__ uchar FCB_FetchEntry_DirtyBit(FileSystem* fs, u32 FCB_index)
{
	// byte[21], bit 7
	return (*(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 21) >> 7);
}
__device__ uchar FCB_FetchEntry_DirectoryBit(FileSystem* fs, u32 FCB_index)
{
	// bit [30]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 37);
}
__device__ uchar FCB_FetchEntry_Directory1(FileSystem* fs, u32 FCB_index)
{
	// bit [37]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 32);
}
__device__ uchar FCB_FetchEntry_Directory2(FileSystem* fs, u32 FCB_index)
{
	// bit [33]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 33);
}
__device__ uchar FCB_FetchEntry_Directory3(FileSystem* fs, u32 FCB_index)
{
	// bit [34]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 34);
}
__device__ uchar FCB_FetchEntry_DirectoryMap(FileSystem* fs, u32 FCB_index)
{
	// bit [35]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 35);
}
__device__ uchar FCB_FetchEntry_Depth(FileSystem* fs, u32 FCB_index)
{
	// bit [36]
	return *(fs->volume + fs->SUPERBLOCK_SIZE + FCB_index * fs->FCB_SIZE + 36);
}
// fetch the complete directory information of the entry
// this follow the structure of the directory pointer
__device__ u32 FCB_FetchEntry_Dir_All(FileSystem* fs, u32 FCB_index) 
{
	u32 temp = 0; // temp variable for comparision
	temp |= FCB_FetchEntry_Depth(fs, FCB_index);
	temp |= FCB_FetchEntry_Directory1(fs, FCB_index) << BYTE_OFFSET;
	temp |= FCB_FetchEntry_Directory2(fs, FCB_index) << (BYTE_OFFSET * 2);
	temp |= FCB_FetchEntry_Directory3(fs, FCB_index) << (BYTE_OFFSET * 3);
	return temp;
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

/*
* This function remove a directory from the the file system structure
* If there is direcotry entry in the directory that the current direcotry pointing at, 
* then remove anything under that directory
* This is a recursive funtion
* ----------------------------------------------------------------------
* parameter: FCB_index -> the fcb entry index of the current directory
*/
__device__ void RmDirectory(FileSystem* fs, u32 FCB_index)
{
	if (FCB_FetchEntry_Depth(fs, FCB_index) >= 2) {// base case, directory entry's depth can be at most 2
		return;
	}
	else {// iterate the FCB block
		DIR_Move2Destination(fs, FCB_index); // change the current directory to the directory pointed by directory entry
		u32 currDirInfo = DIR_FetchDirectory_All(fs);
		for (u32 k = 0; k < fs->FCB_ENTRIES; k++) {
			if (FCB_FetchEntry_DirtyBit(fs, k) == 1
				&& FCB_CheckDirectoryMatch_Compare(fs, k, currDirInfo)) { // the entry is in use and the entry is in current directory
				if (FCB_FetchEntry_DirectoryBit(fs, k) == 1) { // the entry is a directory entry
					// update the directory entry to remove the directory
					FCB_UpdateFCBEntry_DirtyBit(fs, k, 0); // free the fcb entry
					RmDirectory(fs, k);
				}
				else if (FCB_FetchEntry_DirectoryBit(fs, k) == 0) { // the entry is file entry
					// remove the file
					Compaction(fs, k);
				}
			}
		}
	}
}

/*
* Set the directory information of the file system
* ---------------------------------------------------------
* parameters: depth: the current depth / directoryX: directory index of direcotry of different depth
*/
__device__ void DIR_SetDirectoryInfo(FileSystem* fs, uchar depth, uchar directory1, uchar directory2, uchar directory3)
{
	*(fs->DIRECTORY) = 0; // reset value
	*(fs->DIRECTORY) |= depth;
	*(fs->DIRECTORY) |= (directory1<<BYTE_OFFSET);
	*(fs->DIRECTORY) |= (directory2 << (BYTE_OFFSET*2));
	*(fs->DIRECTORY) |= (directory3 << (BYTE_OFFSET*3));
}
// get the directory index of a certain depth
__device__ uchar DIR_FetchDirectory(FileSystem* fs, int depth)
{
	if (depth == 1) {
		return (((*(fs->DIRECTORY)) >> BYTE_OFFSET) & 0xFF);
	}
	else if (depth == 2) {
		return (((*(fs->DIRECTORY)) >> (BYTE_OFFSET*2)) & 0xFF);
	}
	else if (depth == 3) {
		return (((*(fs->DIRECTORY)) >> (BYTE_OFFSET*3)) & 0xFF);
	}
}
// get all the directory information of the current directory
__device__ u32 DIR_FetchDirectory_All(FileSystem* fs)
{
	return *(fs->DIRECTORY);
}

// get the depth of the current directory
__device__ uchar DIR_FetchDepth(FileSystem* fs)
{
	return (*(fs->DIRECTORY) & 0xFF);
}

// get thhe dirInfo of the parent directory of the current directory
__device__ u32 DIR_GetParentDirInfo(FileSystem* fs) 
{
	// 1. get the depth of the current directory
	uchar currDepth = DIR_FetchDepth(fs);
	u32 temp = 0;
	// 2. get the parent directory info according to the depth
	if (currDepth == 0) {
		printf("this is the root directory!\n");
		return 0;
	}
	else if (currDepth <= 1) {
		temp = 0;
	}
	else if (currDepth == 2) {
		temp |= DIR_FetchDirectory(fs, 1);
	}
	else if (currDepth == 3) {
		temp |= DIR_FetchDirectory(fs, 1);
		temp |= (DIR_FetchDirectory(fs, 2)<<BYTE_OFFSET);
	}
	else {
		printf("Invalid directory!\n");
		return 0;
	}
	return temp;
}

/*
* The following 2 functions will move the current directory
* Dir_Move2Parent: move the the parent directory
*/
__device__ void DIR_Move2Parent(FileSystem* fs) 
{
	// 1. get the depth and directory
	uchar currDepth = DIR_FetchDepth(fs);
	// 2. update the current directory: clear deepest dirIndex, depth--
	if (currDepth == 0) {
		printf("cannot go to parent directory, this is the root directory\n");
		return;
	}
	else if (currDepth == 1) DIR_SetDirectoryInfo(fs, 0, 0, 0, 0);
	else if (currDepth == 2) DIR_SetDirectoryInfo(fs, currDepth - 1, DIR_FetchDirectory(fs, 1), 0, 0);
	else if (currDepth == 3) DIR_SetDirectoryInfo(fs, currDepth - 1,
		DIR_FetchDirectory(fs, 1), DIR_FetchDirectory(fs, 2), 0);
	else {
		printf("Invalid directory!\n");
		return;
	}
}
/*
* The following 2 functions will move the current directory
* Dir_Move2Destination: move the the desired directory
* -------------------------------------------------------------
* parameters: FCB_index: the FCB entry index of the destination directory
* note: the desired directory is pointed by the FCB entry sepcified by the FCB_index
*/
__device__ void DIR_Move2Destination(FileSystem* fs, u32 FCB_index) 
{
	uchar dirDepth = DIR_FetchDepth(fs);
	if (dirDepth >= 3) {
		printf("cannot enter directory due to the directory depth limit\n");
		return;
	}
	// get the dirmap of the directory entry
	uchar dirMap = FCB_FetchEntry_DirectoryMap(fs, FCB_index);
	uchar currDepth = DIR_FetchDepth(fs); // depth of current direcotry
	// update the current directory by setting the dirmap in the fs->DIRECTORY pointer
	uchar dir1, dir2, dir3;
	if (currDepth == 0) {
		dir2 = dir3 = 0;
		dir1 = dirMap;
	}
	else if (currDepth == 1) {
		dir1 = DIR_FetchDirectory(fs, 1);
		dir2 = dirMap;
		dir3 = 0;
	}
	else if (currDepth == 2) {
		dir1 = DIR_FetchDirectory(fs, 1);
		dir2 = DIR_FetchDirectory(fs, 2);
		dir3 = 0;
	}
	else {
		printf("Invalid directory!\n");
		return;
	}
	DIR_SetDirectoryInfo(fs, currDepth + 1, dir1, dir2, dir3);
}

/*
* The function find the parent directory of the file
* -----------------------------------------------------------
* parameter: FCB_index: the fcb index of the current file
* return: u32 dirInfo: the information of the parent directory
*/
__device__ u32 DIR_FileGetParent(FileSystem* fs, u32 FCB_index)
{
	u32 dirInfo = FCB_FetchEntry_Dir_All(fs, FCB_index); // directory info of the file
	uchar fileDepth = FCB_FetchEntry_Depth(fs, FCB_index);
	u32 temp = 0;
	// if depth <= 1, the parent is root, and dir info should be 0
	if (fileDepth == 2) {
		temp |= FCB_FetchEntry_Directory1(fs, FCB_index);
	}
	else if (fileDepth == 3) {
		temp |= FCB_FetchEntry_Directory1(fs, FCB_index);
		temp |= (FCB_FetchEntry_Directory2(fs, FCB_index) << BYTE_OFFSET);
	}
	return temp;
}

/*
* The function find the directory entry of the current file
* -----------------------------------------------------------
* parameter: FCB_index: the fcb index of the current file
* return: the FCB_index of the directory entry 
*/
__device__ u32 DIR_FileGetDir(FileSystem* fs, u32 FCB_index)
{
	u32 dirInfo = DIR_FileGetParent(fs, FCB_index);
	uchar fileDepth = FCB_FetchEntry_Depth(fs, FCB_index);
	uchar dirMap;
	if (fileDepth == 1) {
		dirMap = FCB_FetchEntry_Directory1(fs, FCB_index);
	}
	else if (fileDepth == 2) {
		dirMap = FCB_FetchEntry_Directory2(fs, FCB_index);
	}
	else if (fileDepth == 3) {
		dirMap = FCB_FetchEntry_Directory3(fs, FCB_index);
	}
	// find the directory entry accordind to the dirmap and the dirInfo
	u32 dir_fcbIndex = FCB_FindDirectoryIndex_DirInfo(fs, dirInfo, dirMap);
	return dir_fcbIndex;
}


/*
* The function will count the length of a string
*/
__device__ int STR_Length(char* s)
{
	int count = 0;
	while (s[count] != '\0') {
		count++;
	}
	count++;
	return count;
}
