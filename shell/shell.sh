#!/bin/bash
################shell变量###########
#1、使用一个定义过的变量，只要在变量名前面加美元符号即可,变量名和等号之间不能有空格
your_name="zhuxiaoxiao"
echo $your_name
echo ${your_name}

#2、只读变量，使用readonly命令可以将变量定义为只读变量只读变量的值不能被改变
#下面的例子尝试更改只读变量，结果报错：
myUrl="http://www.google.com"
readonly myUrl
myUrl="http://www.runoob.com"

#3、删除变量
#使用unset命令可以删除变量，变量被删除后不能再次使用。unset命令不能删除只读变量
myUrl="http://www.runoob.com"
unset myUrl

#####################shell字符串################
#1、双引号的优点：双引号里可以有变量，双引号里可以出现转义字符
your_name='runoob'
str="Hello, I know you are \"$your_name\"! \n"
echo $str
#输出结果：Hello, I know you are "runoob"!

#2、拼接字符串
your_name="runoob"
#使用双引号拼接
greeting="hello, "$your_name" !"
greeting_1="hello,${your_name} !"   #两者效果一样
echo $greeting  $greeting_1
#输出结果：hello,runoob!  hello,runoob!

#3、获取字符串长度
string="abcd"
echo ${#string}   #输出4

#4、提取子字符串，以下实例从字符串第2个字符串开始截取4个字符
string="runoob is a great site"
echo ${string:1:4}  # 输出unoo

#5、查找子字符串
#查找字符i或o的位置（哪个字母出现就计算哪个）
string="runoob is a gerat site"
echo `expr index "$string" io`  #输出4

#####################shell数组################
#1、定义数组
#在shell中，用括号来表示数组，数组元素用“空格”符号分割开。定义数组的一般形式为：
#数组名=(值1 值2 值3 值4...)
#例如array_name=(value0 value1 value2 value3)

#2、读取数组
#读取数组元素值得一般格式
# ${数组名[下标]}  ---->例如：valuen=${array_name[index]}
my_array=(A B "C" D)
echo "第一个元素为： ${my_array[0]}"
echo "第二个元素为： ${my_array[1]}"
echo "第三个元素为： ${my_array[2]}"

#3、获取数组的长度:方法与获取字符串长度的方法相同
#取得数组元素的个数
length=${#array_name[@]}
# 或者
length=${#array_name[*]}
#取得数组单个元素的长度
lengthn=${#array_name[n]}

#4、获取数组中的所有元素：使用@或* 可以获取素组中的所有元素
my_array[0]=A
my_array[1]=B
my_array[2]=C
my_array[3]=D
echo "数组的元素为：${my_array[*]}"
echo "数组的元素为：${my_array[@]}"

#example-->遍历数组

#!/bin/bash
my_array=(a b "c","d" abd)
echo "----for 循环遍历输出数组-----"
for i in ${my_array[@]};
do
	echo $i

echo "---::::WHILE循环输出，使用let i++自增::::----"
j=0
while [ $j -lt ${#my_array[@]} ]
do 
	echo ${my_array[$j]}
	let j++
done

echo "---:::WHILE循环输出，使用let "n++"自增：多了双引号，其实不用也可以:::----"
n=0
while [ $n -lt ${#my_array[@]} ]
do
	echo ${my_array[$n]}
	let "n++"
done

echo "-----::::WHILE循环输出，使用let m+=1 自增，这种写法其他编程中也常用:::----"
m=0
while [ $m -lt ${#my_array[@]} ]
do
	echo ${my_array[$m]}
	let m+=1
done

echo "----::WHILE循环输出，使用 a=$[$a+1]自增，个人觉得这种写法比较麻烦::::-----"
a=0
while [ $a -lt ${#my_array[@]} ]
do
	echo ${my_array[$a]}
	a=$[$a+1]
done




##########################shell基本算法运算符################
#shell和其他编程语言一样，支持多种运算符，包括：
#算数运算符、关系运算符、布尔运算符、字符串运算符、文件测试运算符
#1、算数运算符实例
#！/bin/bash
echo "*********simple math in shell**********"
a=10
b=20

d=$((a+b)) #等同于d=`expr $a + $b`
e=$((a-b))
f=$((a*b)) #等同于f=`expr $a \* $b`
g=$((a/b)) #等同于g=`exper $a / $b`
h=$((b%a))

echo $b
echo d
echo e
echo f


#2. 关系运算符
# -eq -->  检测两个数是否相等，相等返回true
# -ne -->  检测两个数是否不相等，不相等则返回true
# -gt -->  检测左边的数是否大于右边的如果是，则返回true
# -lt -->  检测左边的数是否小于左边的如果是，则返回true
# -ge -->  检测左边的数是否大于等于右边的，如果是，则返回true
# -le -->  检测左边的数是否小于等于右边的，如果是，则返回true

#!/bin/bash
a=10
b=20

if [ $a -eq $b ]
then 
	echo "$a -eq $b: a等于b"
else
	echo "$a -eq $b: a不等于b"
fi

#3. 布尔运算符
# ！ -->非运算，表达式为true则返回false,否则返回true
# -o -->或运算， 有一个表达式为true则返回true
# -a -->与运算， 两个表达式都为true才返回true

#!/bin/bash
a=10
b=20
if [ $a -lt 100 -a $b -gt 15]
then 
	echo "$a小于100且$b 大于15： 返回true"
else
	echo "$a小于100且$b 大于15：返回false"
fi

#4.逻辑运算符
# && -->逻辑的AND  举例 [[$a -lt100 && $b - gt 100]]
# || -->逻辑的OR

#5. 字符串运算符
# = --> 检测两个字符串是否相等，相等返回true
# != --> 检测两个字符串是否相等，不相等返回true
# -z --> 检测字符串长度是否为0， 为0返回true
# -n --> 检测字符串长度是否为0 ，不为0返回true
# str --> 检测字符串是否为空，不为空返回true


#6. 文件测试运算符
# -r file 检测文件是否可读，如果是，则返回true
# -w file 检测文件是否可写，如果是，则返回true
# -x file 检测文件是否可执行，如果是，则返回true
# -s file 检测文件是否为空（文件大小是否大于0），不为空返回true
# -e file 检测文件是否存在，如果是，则返回true

#====>总结：使用[[...]]条件判断结构，而不是[...],能够防止很多脚本中的许多逻辑错误


###################Shell echo命令###############
#shell的echo命令用于字符串的输出
#1. 显示普通字符串
echo "It is a test"
#2. 显示变量
#read命令从标准输入中读取一行，并把输入行的每个字段的值指定给shell变量
#！/bin/sh
read your_name
echo "$name It is a test"
输入 ok
运行输出：OK It is a test

#总结： read命令一个一个词组地接收输入的参数，每个词组需要使用空格进行分隔；如果输入的词组个数
#大于需要的参数个数，则多出的词组将被作为整体为最后一个参数接收。
#测试文件test.sh代码如下：
read firstStr secondStr
echo "第一个参数：$firstStr; 第二个参数：￥secondStr"
#执行测试
$ sh.test.sh
一 二 三 四
第一个参数：一；第二个参数：二 三 四

#实例，文件test.sh
read -p "请输入一段文字：" -n 6 -t 5 -s password
echo -e "\npassword is $password"
#参数说明：
-p 输入提示文字
-n 输入字符长度限制
-t 输入限时
-s 隐藏输入内容


######################shell printf命令##################
#printf 使用引用文本或空格分隔的参数，外面可以在printf中使用格式化字符串，还可以制定字符串的宽度、
#左右对齐方式等。默认printf不会像echi自动添加换行符，我们可以收到添加\n
#1.printf命令的语法：
printf format-string [ arguments... ]
#参数说明：
format-string: 为格式控制字符串
arguments: 为参数列表

#example
#!/bin/bash
printf "%-10s %-8s %-4s\n" 姓名 性别 体重kg
printf "%-10s %-8s %-4.2f\n" 郭靖 男 66.1234
printf "%-10s %-8s %-4.2f\n" 杨过 男 48.6543
printf "%-10s %-8s 5-4.2f\n" 郭芙 男 47.9876

#%s %c %d %f都是格式代替符
#%-10s 指一个宽度为10个字符（-表示左对齐，没有则表示右对齐），任何字符都会被显示在10个字符宽的字符内，如果不足的自动以空格填充，
#超过也会亮内容全部显示出来
#%-4.2f指格式化为小数，其中.2指保留2位小数
