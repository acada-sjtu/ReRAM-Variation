#include <cstdio>
#include <iostream>
#include <memory.h>
#include <algorithm>    // ʹ�����е� min ����
using namespace std;
const int MAX = 1024;
int n;                // X �Ĵ�С
double weight [MAX] [MAX];        // X �� Y ��ӳ�䣨Ȩ�أ�
double w[5][5]={{3.1,4,6,4,9.2},{6,4,5,3,8},{7,5,3,4,2},{6,3,2,2,5},{8,4,5,4,7}};
double lx [MAX], ly [MAX];        // ���
bool sx [MAX], sy [MAX];    // �Ƿ�������
int match [MAX];        // Y(i) �� X(match [i]) ƥ��
// ��ʼ��Ȩ��
void init (int size);
// �� X(u) Ѱ�������·���ҵ��򷵻� true
bool path (int u);
// ���� maxsum Ϊ true ���������Ȩƥ�䣬������СȨƥ��
double bestmatch (bool maxsum = true);

void init (int size)
{
    // ����ʵ���������Ӵ����Գ�ʼ��
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
    // ��ʼ�����
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
            // �޸ı��
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
                weight [i] [j] = -weight [i] [j];         // �����Ҫ���� weight [ ] [ ] ԭ����ֵ��������Ҫ���仹ԭ
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
//ִ��bestmatch (true) �����Ϊ 29
*/
/*
5
7 6 4 6 1
4 6 5 7 2
3 5 7 6 8
4 7 8 8 5
2 6 5 6 3
//ִ�� bestmatch (false) �����Ϊ 21
*/
