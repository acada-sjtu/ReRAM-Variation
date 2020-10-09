#include <cstdio>
#include <iostream>
#include <memory.h>
#include <algorithm>    // 使用其中的 min 函数
using namespace std;
const int MAX = 1024;
int n;                // X 的大小
double weight [MAX] [MAX];        // X 到 Y 的映射（权重）
double w[5][5]={{3.1,4,6,4,9.2},{6,4,5,3,8},{7,5,3,4,2},{6,3,2,2,5},{8,4,5,4,7}};
double lx [MAX], ly [MAX];        // 标号
bool sx [MAX], sy [MAX];    // 是否被搜索过
int match [MAX];        // Y(i) 与 X(match [i]) 匹配
// 初始化权重
void init (int size);
// 从 X(u) 寻找增广道路，找到则返回 true
bool path (int u);
// 参数 maxsum 为 true ，返回最大权匹配，否则最小权匹配
double bestmatch (bool maxsum = true);

void init (int size)
{
    // 根据实际情况，添加代码以初始化
    n = size;
    for (int i = 0; i < n; i ++)
        for (int j = 0; j < n; j ++)
          //  scanf ("%d", &weight [i] [j]);
          weight[i][j]=w[i][j];
}

bool path (int u)
{
    sx [u] = true;
    for (int v = 0; v < n; v ++)
        if (!sy [v] && lx[u] + ly [v] == weight [u] [v])
            {
            sy [v] = true;
            if (match [v] == -1 || path (match [v]))
                {
                match [v] = u;
                return true;
                }
            }
    return false;
}
double bestmatch (bool maxsum)
{
    int i, j;
    if (!maxsum)
        {
        for (i = 0; i < n; i ++)
            for (j = 0; j < n; j ++)
                weight [i] [j] = -weight [i] [j];
        }
    // 初始化标号
    for (i = 0; i < n; i ++)
        {
        lx [i] = -0x1FFFFFFF;
        ly [i] = 0;
        for (j = 0; j < n; j ++)
            if (lx [i] < weight [i] [j])
                lx [i] = weight [i] [j];
        }
    memset (match, -1, sizeof (match));
    for (int u = 0; u < n; u ++)
        while (1)
            {
            memset (sx, 0, sizeof (sx));
            memset (sy, 0, sizeof (sy));
            if (path (u))
                break;
            // 修改标号
            double dx = 0x7FFFFFFF;
            for (i = 0; i < n; i ++)
                if (sx [i])
                    for (j = 0; j < n; j ++)
                        if(!sy [j])
                            dx = min (lx[i] + ly [j] - weight [i] [j], dx);
            for (i = 0; i < n; i ++)
                {
                if (sx [i])
                    lx [i] -= dx;
                if (sy [i])
                    ly [i] += dx;
                }
            }
    double sum = 0;
    for (i = 0; i < n; i ++)
        sum += weight [match [i]] [i];
    if (!maxsum)
        {
        sum = -sum;
        for (i = 0; i < n; i ++)
            for (j = 0; j < n; j ++)
                weight [i] [j] = -weight [i] [j];         // 如果需要保持 weight [ ] [ ] 原来的值，这里需要将其还原
        }
    return sum;
}

int main()
{
    int n=5;
    init (n);
    double cost = bestmatch (false);
    printf ("%f ", cost);
     cout<<endl;
    for (int i = 0; i < n; i ++)
        {
        printf ("Y %d -> X %d ", i, match [i]);
        cout<<endl;
        }
    return 0;
}
/*
5
3 4 6 4 9
6 4 5 3 8
7 5 3 4 2
6 3 2 2 5
8 4 5 4 7
//执行bestmatch (true) ，结果为 29
*/
/*
5
7 6 4 6 1
4 6 5 7 2
3 5 7 6 8
4 7 8 8 5
2 6 5 6 3
//执行 bestmatch (false) ，结果为 21
*/
