#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[])
{

	/* fork a child process */
	pid_t pid;
	int status;

	printf("Process start to fork\n");
	pid = fork(); // fork a child process

	if (pid==-1) // fork failed
	{
		perror("fork");
		exit(1);
	}
	else
	{
		if (pid==0) // child process
		{
			printf("I'm the Child Process, my pid = %d\n", getpid());
			printf("Child process start to execute test program:\n");
			
			/* execute test program */ 
			// create argument array for the execve
			int i;
            char *arg[argc];
            for(i=0;i<argc-1;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;
			
			// do execve
			execve(arg[0], arg, NULL);
			// output error if running the orignal process
			printf("Continue to run the original process.\n");
			perror("execve");
			exit(EXIT_FAILURE);
		}
		else // parent process
		{
			printf("I'm the Parent Process, my pid = %d\n", getpid());

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receiving the SIGCHLD signal\n");

			/* check child process'  termination status */
			if(WIFEXITED(status))
			{
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status))
			{
				if (WTERMSIG(status) == 6)
				{
					printf("child process get SIGABRT signal\n");
					printf("child process encounters abnormal termination\n");
				}
				else if (WTERMSIG(status) == 14)
				{
					printf("child process get SIGALRM signal\n");
					printf("child process is alarmed by the real-timeclock\n");
				}
				else if (WTERMSIG(status) == 7)
				{
					printf("child process get SIGBUS signal\n");
					printf("child process encounters bus error\n");
				}
				else if (WTERMSIG(status) == 8)
				{
					printf("child process get SIGFPE signal\n");
					printf("child process encounters floating-point error\n");
				}
				else if (WTERMSIG(status) == 1)
				{
					printf("child process get SIGHUP signal\n");
					printf("child process is hung up\n");
				}
				else if (WTERMSIG(status) == 4)
				{
					printf("child process get SIGILL signal\n");
					printf("child process encounters illegal instruction\n");
				}
				else if (WTERMSIG(status) == 2)
				{
					printf("child process get SIGINT signal\n");
					printf("child process gets interrupt from keyboard\n");
				}
				else if (WTERMSIG(status) == 9)
				{
					printf("child process get SIGKILL signal\n");
					printf("child process encounters forced process termination\n");
				}
				else if (WTERMSIG(status) == 13)
				{
					printf("child process get SIGPIPE signal\n");
					printf("child process writes to pipe with no reader\n");
				}
				else if (WTERMSIG(status) == 3)
				{
					printf("child process get SIGQUIT signal\n");
					printf("child process quits\n");
				}
				else if (WTERMSIG(status) == 11)
				{
					printf("child process get SIGSEGV signal\n");
					printf("child process refers to invalid memory\n");
				}
				else if (WTERMSIG(status) == 15)
				{
					printf("child process get SIGTERM signal\n");
					printf("child process encounters breaking point for debugging\n");
				}
				else if (WTERMSIG(status) == 5)
				{
					printf("child process get SIGTRAP signal\n");
					printf("child process is trapped\n");
				}
                printf("CHILD EXECUTION FAILED\n");
            }
            else if(WIFSTOPPED(status))
			{
				printf("child process get SIGSTOP signal\n");
				printf("child process stopped\n");
                printf("CHILD PROCESS STOPPED\n");
            }
            else
			{
                printf("CHILD PROCESS CONTINUED\n");
            }
			exit(1);
		}

	}
	
	
}
