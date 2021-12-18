#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(FileSystem *fs, uchar *input, uchar *output) {
	
	
	/////////////// Test Case 1  ///////////////
	u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs,LS_D);
	fs_gsys(fs, LS_S);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 64, 12, fp);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, RM, "t.txt\0");
	fs_gsys(fs, LS_S);

	
	/////////////// Test Case 2  ///////////////
	/*u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs,input, 64, fp);
	fp = fs_open(fs,"b.txt\0", G_WRITE);
	fs_write(fs,input + 32, 32, fp);
	fp = fs_open(fs,"t.txt\0", G_WRITE);
	fs_write(fs,input + 32, 32, fp);
	fp = fs_open(fs,"t.txt\0", G_READ);
	fs_read(fs,output, 32, fp);
	fs_gsys(fs,LS_D);
	fs_gsys(fs,LS_S);
	fp = fs_open(fs,"b.txt\0", G_WRITE);
	fs_write(fs,input + 64, 12, fp);
	fs_gsys(fs,LS_S);
	fs_gsys(fs,LS_D);
	fs_gsys(fs,RM, "t.txt\0");
	fs_gsys(fs,LS_S);
	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = fs_open(fs,fname[i], G_WRITE);
		fs_write(fs,input + i, 24 + i, fp);
	}

	fs_gsys(fs,LS_S);

	for (int i = 0; i < 5; i++)
		fs_gsys(fs,RM, fname[i]);

	fs_gsys(fs,LS_D);*/
	
	
	/////// Test Case 3  ///////////////
	/*u32 fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input, 64, fp);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_WRITE);
	fs_write(fs, input + 32, 32, fp);
	fp = fs_open(fs, "t.txt\0", G_READ);
	fs_read(fs, output, 32, fp);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, LS_S);
	fp = fs_open(fs, "b.txt\0", G_WRITE);
	fs_write(fs, input + 64, 12, fp);
	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);
	fs_gsys(fs, RM, "t.txt\0");
	fs_gsys(fs, LS_S);

	char fname[10][20];
	for (int i = 0; i < 10; i++)
	{
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}

	for (int i = 0; i < 10; i++)
	{
		fp = fs_open(fs, fname[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);

	for (int i = 0; i < 5; i++)
		fs_gsys(fs, RM, fname[i]);

	fs_gsys(fs, LS_D);

	char fname2[1018][20];
	int p = 0;

	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++)
		{
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}

	for (int i = 0; i < 1001; i++)
	{
		fp = fs_open(fs, fname2[i], G_WRITE);
		fs_write(fs, input + i, 24 + i, fp);
	}

	fs_gsys(fs, LS_S);
	fp = fs_open(fs, fname2[1000], G_READ);
	fs_read(fs, output + 1000, 1024, fp);

	char fname3[17][3];
	for (int i = 0; i < 17; i++)
	{
		fname3[i][0] = 97 + i;
		fname3[i][1] = 97 + i;
		fname3[i][2] = '\0';
		fp = fs_open(fs, fname3[i], G_WRITE);
		fs_write(fs, input + 1024 * i, 1024, fp);
	}

	fp = fs_open(fs, "EA\0", G_WRITE);
	fs_write(fs, input + 1024 * 100, 1024, fp);
	fs_gsys(fs, LS_S);*/

	/////////////// Test Case 4  ///////////////
	// this test case is provided by one of my classmates :)
	/*u32 fp;
	char fname2[1018][20];
	int p = 0;

	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++)
		{
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}

	for (int i = 0; i < 1001; i++)
	{
		fp = fs_open(fs, fname2[i], G_WRITE);
		fs_write(fs, input + 1024 * i, 1024, fp);
	}

	fp = fs_open(fs, "AAA\0", G_WRITE);
	fs_write(fs, input + 1024 * 1001, 1024, fp);
	fp = fs_open(fs, "BBB\0", G_WRITE);
	fs_write(fs, input + 1024 * 1002, 1024, fp);
	fp = fs_open(fs, "CCC\0", G_WRITE);
	fs_write(fs, input + 1024 * 1003, 1024, fp);
	fp = fs_open(fs, "DDD\0", G_WRITE);
	fs_write(fs, input + 1024 * 1004, 1024, fp);
	fp = fs_open(fs, "EEE\0", G_WRITE);
	fs_write(fs, input + 1024 * 1005, 1024, fp);
	fp = fs_open(fs, "FFF\0", G_WRITE);
	fs_write(fs, input + 1024 * 1006, 1024, fp);
	fp = fs_open(fs, "GGG\0", G_WRITE);
	fs_write(fs, input + 1024 * 1007, 1024, fp);
	fp = fs_open(fs, "HHH\0", G_WRITE);
	fs_write(fs, input + 1024 * 1008, 1024, fp);
	fp = fs_open(fs, "III\0", G_WRITE);
	fs_write(fs, input + 1024 * 1009, 1024, fp);
	fp = fs_open(fs, "JJJ\0", G_WRITE);
	fs_write(fs, input + 1024 * 1010, 1024, fp);
	fp = fs_open(fs, "KKK\0", G_WRITE);
	fs_write(fs, input + 1024 * 1011, 1024, fp);
	fp = fs_open(fs, "LLL\0", G_WRITE);
	fs_write(fs, input + 1024 * 1012, 1024, fp);
	fp = fs_open(fs, "MMM\0", G_WRITE);
	fs_write(fs, input + 1024 * 1013, 1024, fp);
	fp = fs_open(fs, "NNN\0", G_WRITE);
	fs_write(fs, input + 1024 * 1014, 1024, fp);
	fp = fs_open(fs, "OOO\0", G_WRITE);
	fs_write(fs, input + 1024 * 1015, 1024, fp);
	fp = fs_open(fs, "PPP\0", G_WRITE);
	fs_write(fs, input + 1024 * 1016, 1024, fp);
	fp = fs_open(fs, "QQQ\0", G_WRITE);
	fs_write(fs, input + 1024 * 1017, 1024, fp);
	fp = fs_open(fs, "RRR\0", G_WRITE);
	fs_write(fs, input + 1024 * 1018, 1024, fp);
	fp = fs_open(fs, "SSS\0", G_WRITE);
	fs_write(fs, input + 1024 * 1019, 1024, fp);
	fp = fs_open(fs, "TTT\0", G_WRITE);
	fs_write(fs, input + 1024 * 1020, 1024, fp);
	fp = fs_open(fs, "UUU\0", G_WRITE);
	fs_write(fs, input + 1024 * 1021, 1024, fp);
	fp = fs_open(fs, "VVV\0", G_WRITE);
	fs_write(fs, input + 1024 * 1022, 1024, fp);
	fp = fs_open(fs, "WWW\0", G_WRITE);
	fs_write(fs, input + 1024 * 1023, 1024, fp);

	fs_gsys(fs, LS_S);
	fs_gsys(fs, LS_D);


	for (int i = 0; i < 1001; i++)
	{
		fp = fs_open(fs, fname2[i], G_READ);
		fs_read(fs, output + 1024 * i, 1024, fp);
	}

	fp = fs_open(fs, "AAA\0", G_READ);
	fs_read(fs, output + 1024 * 1001, 1024, fp);
	fp = fs_open(fs, "BBB\0", G_READ);
	fs_read(fs, output + 1024 * 1002, 1024, fp);
	fp = fs_open(fs, "CCC\0", G_READ);
	fs_read(fs, output + 1024 * 1003, 1024, fp);
	fp = fs_open(fs, "DDD\0", G_READ);
	fs_read(fs, output + 1024 * 1004, 1024, fp);
	fp = fs_open(fs, "EEE\0", G_READ);
	fs_read(fs, output + 1024 * 1005, 1024, fp);
	fp = fs_open(fs, "FFF\0", G_READ);
	fs_read(fs, output + 1024 * 1006, 1024, fp);
	fp = fs_open(fs, "GGG\0", G_READ);
	fs_read(fs, output + 1024 * 1007, 1024, fp);
	fp = fs_open(fs, "HHH\0", G_READ);
	fs_read(fs, output + 1024 * 1008, 1024, fp);
	fp = fs_open(fs, "III\0", G_READ);
	fs_read(fs, output + 1024 * 1009, 1024, fp);
	fp = fs_open(fs, "JJJ\0", G_READ);
	fs_read(fs, output + 1024 * 1010, 1024, fp);
	fp = fs_open(fs, "KKK\0", G_READ);
	fs_read(fs, output + 1024 * 1011, 1024, fp);
	fp = fs_open(fs, "LLL\0", G_READ);
	fs_read(fs, output + 1024 * 1012, 1024, fp);
	fp = fs_open(fs, "MMM\0", G_READ);
	fs_read(fs, output + 1024 * 1013, 1024, fp);
	fp = fs_open(fs, "NNN\0", G_READ);
	fs_read(fs, output + 1024 * 1014, 1024, fp);
	fp = fs_open(fs, "OOO\0", G_READ);
	fs_read(fs, output + 1024 * 1015, 1024, fp);
	fp = fs_open(fs, "PPP\0", G_READ);
	fs_read(fs, output + 1024 * 1016, 1024, fp);
	fp = fs_open(fs, "QQQ\0", G_READ);
	fs_read(fs, output + 1024 * 1017, 1024, fp);
	fp = fs_open(fs, "RRR\0", G_READ);
	fs_read(fs, output + 1024 * 1018, 1024, fp);
	fp = fs_open(fs, "SSS\0", G_READ);
	fs_read(fs, output + 1024 * 1019, 1024, fp);
	fp = fs_open(fs, "TTT\0", G_READ);
	fs_read(fs, output + 1024 * 1020, 1024, fp);
	fp = fs_open(fs, "UUU\0", G_READ);
	fs_read(fs, output + 1024 * 1021, 1024, fp);
	fp = fs_open(fs, "VVV\0", G_READ);
	fs_read(fs, output + 1024 * 1022, 1024, fp);
	fp = fs_open(fs, "WWW\0", G_READ);
	fs_read(fs, output + 1024 * 1023, 1024, fp);

	//fs_gsys(fs, RM, "2A\0");


	fp = fs_open(fs, "2A\0", G_WRITE);
	fs_gsys(fs, LS_S);
	fs_write(fs, input, 1022, fp);
	fs_gsys(fs, LS_D);*/
}
