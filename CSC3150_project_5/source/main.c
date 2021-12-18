#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include <linux/uaccess.h>
#include "ioc_hw5.h"
#include "linux/irqreturn.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"


int* dev_id = NULL;

// character device
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;


// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
#define DMAIRQCOUNTADDR 0x27   // the count of the interrupt
void *dma_buf;


// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;

// Find prime function
int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	
	return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}
/*
* [user] Read operation for the device
* [kernel] Read the result from DMA buffer
* CLean the result and set readable to false afterwards
* usage: read(fd, &ret, sizeof(int));
*/
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	// read result from DMA buffer and put the result to the user mode + clear result
	if (ss == sizeof(int)){
		int value;
		value = myini(DMAANSADDR);
		put_user(value, buffer);
		myouti(0, DMAANSADDR);
		printk("%s:%s(): ans = %d\n",PREFIX_TITLE, __func__, value);
	}
	else if (ss == sizeof(short)){
		short value;
		value = myins(DMAANSADDR);
		put_user(value, buffer);
		myouts(0, DMAANSADDR);
		printk("%s:%s(): ans = %d\n",PREFIX_TITLE, __func__, value);
	}
	else if (ss == sizeof(char)){
		char value;
		value = myinc(DMAANSADDR);
		put_user(value, buffer);
		myoutc(0, DMAANSADDR);
		printk("%s:%s(): ans = %d\n",PREFIX_TITLE, __func__, value);
	}
	// set readable to false
	myouti(0, DMAREADABLEADDR);
	return 0;
}
/*
* [User] Write operation for the device
* [Kernel] Transfer data to the DMA buffer
* Invoke arithmetic in the kernel and put that work in the work queue
* usage: write(fd, &data, sizeof(data));
*/
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	// copy the operation values (dataIn) to the DMA buffer
	//printk("%s() invokes!\n", __func__);
	copy_from_user (dataIn, buffer, sizeof(struct DataIn));
	myoutc((*dataIn).a, DMAOPCODEADDR);
	myouti((*dataIn).b, DMAOPERANDBADDR);
	myouts((*dataIn).c, DMAOPERANDCADDR);
	
	// invoke work routine
	int IOMode; // IO mode: 1: blocking, 0: non-blocking
	IOMode = myini(DMABLOCKADDR); // blocking io is set by IOCTL
	INIT_WORK(work_routine, drv_arithmetic_routine);
	if (IOMode){
		// blocking IO
		printk("%s:%s(): queue work\n",PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		flush_scheduled_work();
	}
	else {
		// non-blocking IO
		printk("%s:%s(): queue work\n",PREFIX_TITLE, __func__);
		schedule_work(work_routine);
	}

	return 0;
}
/*
* Change device configuration
* usage: ioctl(fd, HW5_IOCSETSTUID, &ret)
*/
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	if (cmd == HW5_IOCSETSTUID){ // set student ID
		int value;
		get_user(value, (int*)arg);
		myouti(value, DMASTUIDADDR);
		printk("%s:%s(): My STUID is %d\n", PREFIX_TITLE, __func__, value);
	}
	else if (cmd == HW5_IOCSETRWOK){ // Set if RW OK: printk OK if you complete R/W function
		int value;
		get_user(value, (int*)arg);
		myouti(value, DMARWOKADDR);
		if (value==1) printk("%s:%s(): RM OK\n", PREFIX_TITLE, __func__);
		else printk("%s:%s(): RM NOT OK\n", PREFIX_TITLE, __func__);
	}
	else if (cmd == HW5_IOCSETIOCOK){ // Set if ioctl OK: printk OK if you complete ioctl function
		int value;
		get_user(value, (int*)arg);
		myouti(value, DMAIOCOKADDR);
		if (value==1) printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
		else printk("%s:%s(): IOC NOT OK\n", PREFIX_TITLE, __func__);
	}
	else if (cmd == HW5_IOCSETIRQOK) { // Set if IRQ OK: printk OK if you complete bonus
		int value;
		get_user(value, (int*)arg);
		myouti(value, DMACOUNTADDR);
		myouti(value, DMAIRQOKADDR);
		if (value==1) printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
		else printk("%s:%s(): IRQ NOT OK\n", PREFIX_TITLE, __func__);
	}
	else if (cmd == HW5_IOCSETBLOCK) { // Set blocking or non-blocking: set write function mode
		int value;
		get_user(value, (int*)arg);
		myouti(value, DMABLOCKADDR);
		if (value == 0) printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		else printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
	}
	else if (cmd == HW5_IOCWAITREADABLE) { // Wait if readable now
		while (myini(DMAREADABLEADDR)==0) { // sleep and wait
			msleep(1);
		}
		int value = myini(DMAREADABLEADDR);
		put_user(value, (int*)arg);
		printk("%s:%s(): Wait Readable %d\n", PREFIX_TITLE, __func__, myini(DMAREADABLEADDR));
	}
	else {
		printk("%s:%s(): Invalid command!\n", PREFIX_TITLE, __func__);
	}
	return 0;
}
/*
* execute the computation when the work is in queue
* complete ‘+’, ‘-’, ‘*’, ‘/’ and ‘p’ computations basing on the data being stored in DMA buffer
* check blocking and non-blocking setting from DMA buffer
* non-blocking write, set the readable as true when completed
*/
static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
	//printk("%s() invokes!\n", __func__);
	char a;
	int b;
	short c;
    int ans;
	// load the stored operaion numbers from the DMA buffer
	a = myinc(DMAOPCODEADDR);
	b = myini(DMAOPERANDBADDR);
	c = myins(DMAOPERANDCADDR);

    switch(a) {
        case '+':
            ans=b+c;
            break;
        case '-':
            ans=b-c;
            break;
        case '*':
            ans=b*c;
            break;
        case '/':
            ans=b/c;
            break;
        case 'p':
            ans = prime(b, c);
            break;
        default:
            ans=0;
    }
	printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__, b, a, c, ans);
	// store the answer to the DMA buffer
	myouti(ans, DMAANSADDR);
	// readable set if non-blocking (0)
	if (!myini(DMABLOCKADDR)) myouti(1, DMAREADABLEADDR);
}

/*
* IRQ handler of the keyboard interrupt
* The handler will count the time of the interrupts.
*/
static irqreturn_t countInterrupt(int irq, void *dev_id){
	int temp = myini(DMAIRQCOUNTADDR);
	temp ++;
	myouti(temp, DMAIRQCOUNTADDR);
	//printk("%s:%s(): Add up interrupt count: %d\n", PREFIX_TITLE, __func__, temp);
	return IRQ_HANDLED;
}

static int __init init_modules(void) {
	dev_t dev;
	int ret;
	printk("%s:%s(): ...............Start...............\n", PREFIX_TITLE, __func__);
	dev_cdev = cdev_alloc();
	/* Register chrdev */ 
	if(alloc_chrdev_region(&dev, 0, 1, "mydev") < 0) {
		printk(KERN_ALERT"Register chrdev failed!\n");
		return -1;
    } else {
		printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
    }

    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);

	/* Init cdev and make it alive */
	dev_cdev->ops = &fops;
    	dev_cdev->owner = THIS_MODULE;

	if(cdev_add(dev_cdev, dev, 1) < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
   	}


	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);

	/*add an ISR into an IRQ number's action list*/
	myouti(0, DMAIRQCOUNTADDR);
	dev_id = kmalloc(sizeof(typeof(int)), GFP_KERNEL);
	ret = request_irq(1, countInterrupt, IRQF_SHARED, "mydev", dev_id);
	if (ret<0){
		printk(KERN_ALERT"Request IRQ failed!\n");
	}
	//if (request_irq(1, countInterrupt, IRQF_SHARED, "mydev", dev_id)<0){
	//	printk(KERN_ALERT"Request IRQ failed!\n");
	//}

	/* Allocate work routine */
	dataIn = kmalloc(sizeof(typeof(*dataIn)), GFP_KERNEL);
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	printk("%s:%s(): request_irq 1 returns %d\n", PREFIX_TITLE, __func__, ret);
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);
	return 0;
}

static void __exit exit_modules(void) {
	dev_t dev;
	// print interrupt number
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, myini(DMAIRQCOUNTADDR));
	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major,dev_minor), 1);
	cdev_del(dev_cdev);

	/* Free work routine */
	kfree(dataIn);
	kfree(work_routine);

	/* free interrupt count variable */
	free_irq(1, dev_id);
	kfree(dev_id);

	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
	printk("%s:%s(): ..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
