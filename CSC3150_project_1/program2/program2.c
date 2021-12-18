#include <linux/init.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");
static struct task_struct *task;
struct wait_opts {
	enum pid_type		wo_type;
	int			wo_flags;
	struct pid		*wo_pid;

	struct siginfo __user	*wo_info;
	int __user		*wo_stat;
	struct rusage __user	*wo_rusage;

	wait_queue_t		child_wait;
	int			notask_error;
};

extern long _do_fork(unsigned long clone_flags,
	      unsigned long stack_start,
	      unsigned long stack_size,
	      int __user *parent_tidptr,
	      int __user *child_tidptr,
	      unsigned long tls);
extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);

void my_wait(pid_t pid){
	int status;
	int a;

	struct wait_opts wo;
	struct pid *wo_pid=NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type=type;
	wo.wo_pid=wo_pid;
	wo.wo_flags=WEXITED|WSTOPPED;
	wo.wo_info=NULL;
	wo.wo_stat=(int __user*)&status;
	wo.wo_rusage=NULL;

	a=do_wait(&wo);
	//printk("do_wait return value is %d\n", &a);

	//print out the signals
	if ((*wo.wo_stat&127) == 0) {
		printk("[program2] : child process runs normally");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 6) {
		printk("[program2] : get SIGABRT signal");
		printk("[program2] : child process encounters abnormal termination");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 14) {
		printk("[program2] : get SIGALARM signal");
		printk("[program2] : child process is alarmed by real-timerclock");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 7) {
		printk("[program2] : get SIGBUS signal");
		printk("[program2] : child process encounters bus error");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 8) {
		printk("[program2] : get SIGFPE signal");
		printk("[program2] : child process encounters floating-point exception");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 1) {
		printk("[program2] : get SIGHUP signal");
		printk("[program2] : child process is hung up");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 4) {
		printk("[program2] : get SIGILL signal");
		printk("[program2] : child process encounters illegal instruction");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 2) {
		printk("[program2] : get SIGINT signal");
		printk("[program2] : child process gets interrupt from keyboard");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 9) {
		printk("[program2] : get SIGKILL signal");
		printk("[program2] : child process encounters forced-process termination");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 13) {
		printk("[program2] : get SIGFPIPE signal");
		printk("[program2] : child process writes to pipe with no readers");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 3) {
		printk("[program2] : get SIGQUIT signal");
		printk("[program2] : child process quits");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 11) {
		printk("[program2] : get SIGSEGV signal");
		printk("[program2] : child process refers to invalid memory");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 15) {
		printk("[program2] : get SIGTERM signal");
		printk("[program2] : child process terminates");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 5) {
		printk("[program2] : get SIGTRAP signal");
		printk("[program2] : child process encounters breaking point for debugging");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else if ((*wo.wo_stat&127) == 126) { // ivalid path or file name
		printk("[program2] : invalid path in the my_exec function");
		printk("[program2] : The return signal is %d\n", (*wo.wo_stat&127));
	}
	else {
		printk("[program2] : get SIGSTOP signal");
		printk("[program2] : child process stops");
		printk("[program2] : The return signal is %d\n", 19);
	}

	put_pid(wo_pid); // decrease the count and free memory

	return;
}

// implement the program execution
int my_exec(void){ // excute the test in child process
	int result;
	const char path[]="/opt/test"; // the path of the executed file,(the test)
	const char *const argv[] = {path, NULL, NULL};
	const char *const envp[] = {"HOME=/","PATH=/sbin:/usr/sbin:/bin:/usr/bin",NULL};

	printk("[program2] : child process");
	struct filename * my_filename = getname(path);
	result = do_execve(my_filename, argv, envp);

	if (!result) return 0; // if exec success
	do_exit(result); // if fails
}

//implement fork function
int my_fork(void *argc){
	
	int pid;
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0); // take my_exec as the parameter

	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);
	
	/* execute a test program in child process */
	// the excution is done in the execve
	//printk("[program2] : child process");
	
	/* wait until child process terminates */
	my_wait(pid);
	return 0;
}

static int __init program2_init(void){

	printk("[program2] : module_init\n");
	
	/* write your code here */
	
	/* create a kernel thread to run my_fork */
	//create a kthread
	printk("[program2] : module_init create kthread start\n");
	task=kthread_create(&my_fork,NULL,"MyThread");

	//wake up new thread if ok
	if(!IS_ERR(task)){
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}
	
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);
