#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LOGLENGTH 10           // the base length of the log
#define LOGLENGTHVAR 25        // the variance of log length, the actual log length = LOGLENGTH + range(0, LOGLENGRHVAR)
#define LAG 100000

/* mutexGame: shared data */
#define PLAY 0
#define WIN 1
#define LOSE 2
#define EXIT 3

/* mutexMap: shared data */
int * LOGPOS_ARRAY;          // array to store the starting postion of each log
int * LOGLENGTH_ARRAY;       // array to store the log length
uint8_t GAMESTATUS;          // record current gamestatus
uint8_t BOUND;
char map[ROW+10][COLUMN] ;

/* mutexFrog shared datav*/
struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; // the coordinate of the frog



/*mutex*/
pthread_mutex_t mutexMap;
pthread_mutex_t mutexFrog;
pthread_mutex_t mutexGame;
// note: we need to set lock whether accessing the global variables



// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

/******************************************************************/
/*Control part*/
/*
 * set the game status
 * parameter: the gamestatus index
 * 0: play, 1: win, 2: lose, 3: exit
 */
void control_ChangeGameStatus(int status){
	pthread_mutex_lock(&mutexGame);
	GAMESTATUS = status;
	pthread_mutex_unlock(&mutexGame);
}

/*
 * check the game status
 * return value: true->playmode, false->not playmode
 */
void control_RefreshGameStatus(){
	int frogx, frogy;
	pthread_mutex_lock(&mutexFrog);
	frogx = frog.x;
	frogy = frog.y;
	pthread_mutex_unlock(&mutexFrog);

	pthread_mutex_lock(&mutexMap);
	bool frog_in_river = (map[frogx][frogy] == ' ');
	pthread_mutex_unlock(&mutexMap);
	// forg out of bound?
	if (frogx<0 || frogx>ROW || frogy<0 || frogy>COLUMN-2){
		control_ChangeGameStatus(LOSE);
		pthread_mutex_lock(&mutexGame);
		BOUND = 1;
		pthread_mutex_unlock(&mutexGame);
	}
	// frog to the other bank?
	if (frogx == 0){
		control_ChangeGameStatus(WIN);
	}
	// forg into river?
	if (frog_in_river){
		control_ChangeGameStatus(LOSE);
	}
}


/*
 * check if the current status is playmode
 */
bool control_GameStatusPlaying(){
	pthread_mutex_lock(&mutexGame);
	int currStatus = GAMESTATUS;
	pthread_mutex_unlock(&mutexGame);
	return (currStatus == 0);
}

/******************************************************************/
/*Change map part*/
/*
 * move the frog, only set the x and y of the frog
 * x: move along the column (row change) 
 * y: move along the row (column change)
 * up: -1.0 / down: 1, 0 / left: 0, -1 / right: 0, 1
 */
void changeMap_frog(int x, int y){
	pthread_mutex_lock(&mutexFrog); // access frog, lock mutex
	frog.x += x;
	frog.y += y;
	pthread_mutex_unlock(&mutexFrog);
}

/*
 * change the logs on the map
 * parameters: logid: index of log / start: start index of log / 
 * log: length of log / frogx and frogy, frog position
 * (1) change the log positions in the map
 * (2) if the frog on the log, also change frog position
 */
void changeMap_log(int log_id, int start, int log, int frogx, int frogy){
	int row = log_id+1;
	int length = COLUMN-1;
	int end = (start+log) % length;

	pthread_mutex_lock(&mutexMap);
	// change the map matrix here
	if (start<=end){
		for (int i=0; i<start; i++){
			map[row][i] = ' ';
		}
		for (int i=start; i<end; i++){
			map[row][i] = '=';
		}
		for (int i=end; i<length; i++){
			map[row][i] = ' ';
		}
		// change frog
		if (frogx == row){
			if (frogy>=start && frogy<end){
				if (log_id%2) changeMap_frog(0, 1);
				else changeMap_frog(0, -1);
			}
		}
	}
	else{
		for (int i=0; i<end; i++){
			map[row][i] = '=';
		}
		for (int i=end; i<start; i++){
			map[row][i] = ' ';
		}
		for (int i=start; i<length; i++){
			map[row][i] = '=';
		}
		// change frog
		if (frogx == row){
			if (frogy<end || frogy>=start){
				if (log_id%2) changeMap_frog(0, 1);
				else changeMap_frog(0, -1);
			}
		}
	}
	//map[row][start] = char(log_id+48);
	//map[row][end] = char(log+48);

	pthread_mutex_unlock(&mutexMap);
}

/*
 * Print the whole river map on the screen, only do the print, donnot change the map
 */
void changeMap_PrintRiverMap(){
	char mapTemp[ROW+10][COLUMN];
	Node frogTemp;
	//Print the map into screen
	pthread_mutex_lock(&mutexMap);
	memcpy(mapTemp, map, sizeof(map));
	//for (int m=1; m<ROW; m++){
	//	for (int n=0; n<COLUMN-1; n++){
	//		mapTemp[m][n] = map[m][n];
	//	}
	//}
	pthread_mutex_unlock(&mutexMap);

	pthread_mutex_lock(&mutexFrog);
	frogTemp = frog;
	pthread_mutex_unlock(&mutexFrog);


	for(int j = 0; j < COLUMN - 1; ++j )	
		mapTemp[ROW][j] = map[0][j] = '|' ; // draw the bank

	for(int j = 0; j < COLUMN - 1; ++j )	
		mapTemp[0][j] = map[0][j] = '|' ;
	mapTemp[frogTemp.x][frogTemp.y] = '0'; // draw the frog
	
	printf("\033[H\033[2J");
	for (int i=0; i<ROW+1; i++){
		puts(mapTemp[i]);
	}

	printf("\n"); // for input
}

/******************************************************************/
/*Thread part*/
/*
 * Create a thread to control the movement of one log
 * the thread only changes the frog on the map, since print the map in wach thread seems not w=very efficient
 */
void *thread_logs_move( void *t ){
	long log_id; // should add a while loop
	log_id = (long)t;
	int startIndex = LOGPOS_ARRAY[log_id];
	int logLength = LOGLENGTH_ARRAY[log_id];
    Node tempFrog; // local varibale to store the frog position
	/*  Move the logs  */
	while (control_GameStatusPlaying()){
		if (log_id % 2) { // move right
			startIndex = (startIndex+1+COLUMN-1)%(COLUMN-1);
		}
		else { // move left
			startIndex = (startIndex-1+COLUMN-1)%(COLUMN-1);
		}

		// move the frog on the map
		pthread_mutex_lock(&mutexFrog);
		tempFrog = frog;
		pthread_mutex_unlock(&mutexFrog);
		changeMap_log(log_id, startIndex, logLength, tempFrog.x, tempFrog.y);
		usleep(LAG);
	}
	pthread_exit(NULL);
}

/*
 * thread to control the movement of the frog
 */
void *thread_move_control(void *t){
	while (control_GameStatusPlaying()){
		if (kbhit()){
			char dir = getchar();
            if ( dir == 'w' || dir == 'W' ) changeMap_frog(-1, 0);
            else if ( dir == 's' || dir == 'S' ) changeMap_frog(1, 0);
            else if ( dir == 'a' || dir == 'A' ) changeMap_frog(0, -1);
            else if ( dir == 'd' || dir == 'D' ) changeMap_frog(0, 1);
			else if ( dir == 'q' || dir == 'Q' ) {
                control_ChangeGameStatus(EXIT);
            }
		}
		control_RefreshGameStatus(); // refresh gamestatus after frog move on map
	}
    
	pthread_exit(NULL);
}

/*
 * thread to control the game by 2 means
 * refreshing the map + delay
 */
void *thread_game_control(void *t){
	while (control_GameStatusPlaying()){
		changeMap_PrintRiverMap();
		usleep(LAG);
	}
	pthread_exit(NULL);
}


/******************************************************************/
/*main program*/
int main( int argc, char *argv[] ){
	/************************************************?
	/* Initialize flags and map */
	
	GAMESTATUS = 0; // playmode
	BOUND = 0;
	
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  // fill the river
	}	
	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ; // drwa the bank

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;
	
	// initialize frog
	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	// initialize log positions and lengths
	srand(time(NULL));
	LOGPOS_ARRAY = (int*)malloc(sizeof(int)*(ROW-1));
	for (int i=0; i<ROW-1; i++){
		LOGPOS_ARRAY[i] = rand()%(COLUMN-1);
	}

	LOGLENGTH_ARRAY = (int*)malloc(sizeof(int)*(ROW-1));
	for (int i=0; i<ROW-1; i++){
		LOGLENGTH_ARRAY[i] = LOGLENGTH + rand()%LOGLENGTHVAR;
	}

	for (int i=0; i<ROW-1; i++){
		int start = LOGPOS_ARRAY[i];
		int len = LOGLENGTH_ARRAY[i];
		for (int k=0; k<len; k++){
			map[i+1][(start+k+(COLUMN-1))%(COLUMN-1)] = '=';
		}
	}

	//Print the map into screen
	printf("\033[H\033[2J");
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );
	printf("\n"); // for input


    /************************************************?
	/* Initialization regarding the threads  */
	pthread_mutex_init(&mutexMap, NULL);
	//pthread_mutex_init(&mutexFrog, NULL);
	pthread_mutex_init(&mutexGame, NULL);
	long id;

	/* Create pthreads and synchronize the threads*/
	
	pthread_t th[ROW+1];
	
	// log threads
	for (id=0; id<ROW-1; id++){
		if (pthread_create(&th[id], NULL, thread_logs_move, (void*)id) != 0){
			perror("Failed to create thread");
		}
	}
	
	// frog move control thread
	if (pthread_create(&th[ROW-1], NULL, thread_move_control, NULL) != 0){
		perror("Failed to create thread");
	}
	
	// game status control thread
	if (pthread_create(&th[ROW], NULL, thread_game_control, NULL) != 0){
		perror("Failed to create thread");
	}

	// sychronize processes
	for (id=0; id<ROW-1; id++){
		if (pthread_join(th[id], NULL) != 0){
			perror("Fail to join thread");
		}
	}
	if (pthread_join(th[ROW-1], NULL) != 0){
		perror("Fail to join thread");
	}
	if (pthread_join(th[ROW], NULL) != 0){
		perror("Fail to join thread");
	}
	
	/************************************************?
	/*  Display the output for user: win, lose or quit.  */
	if (!BOUND){ // out of bound will not print the terminate info
		changeMap_PrintRiverMap();
		usleep(3*LAG);
	}
	printf("\033[?25h\033[H\033[2J");
	switch (GAMESTATUS)
	{
	case WIN:
		printf("You win the game! :)\n");
		break;
	case LOSE:
		printf("You lose the game! :(\n");
		break;
	case EXIT:
		printf("You quit the game!\n");
		break;
	default:
		break;
	}

	/* Destroy and exit calls regarding pthread */
	pthread_mutex_destroy(&mutexMap);
	pthread_mutex_destroy(&mutexGame);
	free(LOGPOS_ARRAY);
	pthread_exit(NULL);
	return 0;

}
