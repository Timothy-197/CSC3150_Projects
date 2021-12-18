#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <signal.h>

// mem: shared memory, size: num of processes, iter: iteration index, arg: arguments
void fork_child_process(int* mem, int size, int iter, char* arg[]){
	if (iter==0) return; // base case
	// fork a child
	pid_t pid = fork();

	if (pid == -1) {
		perror("fork");
		exit(1);
	}
	else {
		if (pid == 0) { // child
			mem[size+size-iter+1] = getpid();
			fork_child_process(mem, size, iter-1, arg);
			execve(arg[size-iter], NULL, NULL);
			// if proceed, there should be error
			perror("execve");
			exit(1);
		}
		else { // parent
			if (iter==size){ // the initial parent process
				mem[size] = getpid();
			}
			// wait for signal
			int wstatus;
			pid_t child_pid = wait(&wstatus);
			if (child_pid == -1) abort();
			mem[size-iter] = wstatus;
		}
	}
}

// info: info of pids of processes, signals, size is the number of processes
void printProcessInfo(int* info, int size){
	printf("---\n");
	printf("process tree: %d", info[size-1]);
	for (int i=1; i<size; i++){
		printf("->%d", info[i+size-1]);
	}
	printf("\n");

	for (int i=0; i<size-1; i++){
		int status = info[size-2-i];
		if(WIFEXITED(status)){
            printf("Child process %d of parent process %d terminated normally with exit code %d\n", info[2*size-2-i], info[2*size-3-i], WEXITSTATUS(status));
        }
        else if(WIFSIGNALED(status)){
            printf("Child process %d of parent process %d is terminated by signal %d", info[2*size-2-i], info[2*size-3-i], WTERMSIG(status));
			// print out corresponding signals
			if (WTERMSIG(status) == 6)
				{
					printf("(Abort)\n");
				}
				else if (WTERMSIG(status) == 14)
				{
					printf("(Alarm)\n");
				}
				else if (WTERMSIG(status) == 7)
				{
					printf("(Bus)\n");
				}
				else if (WTERMSIG(status) == 8)
				{
					printf("(Floating)\n");
				}
				else if (WTERMSIG(status) == 1)
				{
					printf("(Hangup)\n");
				}
				else if (WTERMSIG(status) == 4)
				{
					printf("(Illegal_instr)\n");
				}
				else if (WTERMSIG(status) == 2)
				{
					printf("(Interrupt)\n");
				}
				else if (WTERMSIG(status) == 9)
				{
					printf("(Kill)\n");
				}
				else if (WTERMSIG(status) == 13)
				{
					printf("(Pipe)\n");
				}
				else if (WTERMSIG(status) == 3)
				{
					printf("(Quit)\n");
				}
				else if (WTERMSIG(status) == 11)
				{
					printf("(Segment Fault)\n");
				}
				else if (WTERMSIG(status) == 15)
				{
					printf("(Terminate)\n");
				}
				else if (WTERMSIG(status) == 5)
				{
					printf("(Trap)\n");
				}
        }
        else if(WIFSTOPPED(status)){
            printf("Child process %d of parent process %d is terminated by signal %d (Stop)\n", info[2*size-2-i], info[2*size-3-i], WIFSTOPPED(status));
        }
        else{
			printf("something is wrong\n");
			exit(1);
        }
	}
	printf("Myfork process (%d) terminated normally\n", info[size-1]);
}

int main(int argc, char *argv[]){
	// read arguments
	int i;
    char *arg[argc];
    for(i=0;i<argc-1;i++){
        arg[i]=argv[i+1];
    }
    arg[argc-1]=NULL;

	// create shared memory
	int * sharedMem = mmap(NULL, (2*argc-1)*sizeof(int), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);

	// recusively create child process
	fork_child_process(sharedMem, argc-1, argc-1, arg);
	
	//for (int i=0; i<(argc-1)*2+1; i++){ // a code to check the sharedMem, just for debugging
	//	printf("the SIGs: %d\n", sharedMem[i]);
	//}
	printProcessInfo(sharedMem, argc);

	exit(1);
}
