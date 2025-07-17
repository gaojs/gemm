/* 
In the test driver, there is a loop "for ( p=PFIRST; p<= PLAST; p+= PINC )"
The below parameters set this range of values that p takes on 
在测试驱动程序中，有一个循环“for（p=PFIRST；p<=PLAST；p+=PINC）”
以下参数设置了p的取值范围
*/   
#define PFIRST 128
#define PLAST  2048
#define PINC   128

/* 
In the test driver, the m, n, and k dimensions are set to the below 
values.  If the value equals "-1" then that dimension is bound to the
index p, given above.
在测试驱动器中，m、n和k尺寸设置为以下值。如果该值等于“-1”，则该维度绑定到上面给出的索引p。
*/

#define M -1
#define N -1
#define K -1

/* 
In the test driver, each experiment is repeated NREPEATS times and
the best time from these repeats is used to compute the performance
在测试驱动器中，每个实验重复NREPEATS次，并使用这些重复中的最佳时间来计算性能
*/

#define NREPEATS 3

/* 
Matrices A, B, and C are stored in two dimensional arrays with
row dimensions that are greater than or equal to the row dimension
of the matrix.  This row dimension of the array is known as the 
"leading dimension" and determines the stride (the number of 
double precision numbers) when one goes from one element in a row
to the next.  Having this number larger than the row dimension of
the matrix tends to adversely affect performance.  LDX equals the
leading dimension of the array that stores matrix X.  If LDX=-1 
then the leading dimension is set to the row dimension of matrix X.
矩阵A、B和C存储在二维数组中，其行维度大于或等于矩阵的行维度。数组的行维度称为前导维度，
它决定了从一行中的一个元素到下一个元素的步幅（双精度数字的数量）。如果这个数字大于矩阵的行维，
往往会对性能产生不利影响。LDX等于存储矩阵X的数组的前导维度。
如果LDX=-1，则前导维度设置为矩阵X的行维。
*/

#define LDA -1
#define LDB -1
#define LDC -1
