#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "math.h"
#include <fstream>
#include <time.h>
#include <string>
#include <ctime>
#include <windows.h>
//#include <z3.h>
#include "test_capi.h"
using namespace std;

//#define _FILENET_
//#define _FILEFLOW_
//#define _SCHORIGINAL_
//#define _AQDEBUG_
//#define _LOG_Z3_CALLS_
//#define _FILEDELAY_ 
//#define _DEBUG_
//#define _EDFFEX_
//#define _EDFFEXWQ_
//#define _FPEDF_
//#define _MY_

#define _NWEDF_
//#define _Z3_

#define N 40 // SW + SW * ES
#define SW 20
#define ES 1 // each SW has ES nodes
#define Q 4
#define F 10000//10
#define NUMPORT 4
#define PEUP 9 //6
#define PELOW 6
#define K 16 //8 // pow2[PEUP-PELOW+1]
#define PEUNIT 64 //PEUNIT*PELOW > (SW+2)*SIZEUP ��8ʱ�����㷨������
#define H  32768 //640 //pow2[PEUP] * PEUNIT;
#define MAXA 32768
#define SIZEUP 7 //
#define SIZELOW 1
#define BJ 5
#define L 20
//#define TH 9
//**********�����Ե� 9SW100Q8F10000L8PELOW7PEUP9PEUNIT64SIZELOW1SIZEUP2BJ5
//һ����Ч��Z3�ɼ���� 20 10 1 2 100/10 4 8 5 5 4 4 8 256 256 8 1


#define REFILE sprintf_s(refilename,"SW%dQ%dF%dPELOW%dPEUP%dPEUNIT%dSIZELOW%dSIZEUP%dBJ%dL%d.txt",SW,Q,F,PELOW,PEUP,PEUNIT,SIZELOW,SIZEUP,BJ,L);

#define MAX 9999999
#define NULL 0
#define SRE 1 //release
#define SBL 2 //block
#define SFI 3 //finish
#define ING 4 //is transmitting


#define CD 40
#define A ((int)(1324*SW))
#define EL ((int)sqrt((double)A))

struct node
{
	int nid;
	int x;
	int y;
	int jointed;
	int neighbor;
	int type; //1 - SW; 2 - ES
};

struct flow
{
	int s;
	int d;
	int c;
	int size;
	int period;
	int deadline;
	int pathNode[N];
	int pathPort[N];
	int qid;
};

struct link
{
	int a;
	int sp;
	int dp;
};

struct sortpacket
{
	int f;
	int k;
	int release;
	int deadline;
};

struct node nd[N];
struct flow fl[F];
struct link lk[N][N];
int curwl[H][SW][NUMPORT+1][Q];
int qC[F];
int aC,bC;
int numofpacket;
int tCUp[SW];
int mys[N][NUMPORT+1][H];
int FexEDFEn,FPEn,MyEn,Z3En,ORGEn;

int numofas;
int sumoftableSMT;
int pow2[11] = {1,2,4,8,16,32,64,128,256,512,1024};

int delayNWEDF[F][K];
int delayNWEDFFexWQ[F][K];
int delayMy[F][K];
int myEntries[N];
int delayNWEDFAna[F][K];
int releaseTime[F][K],deadlineTime[F][K],mu1[F][F],mu2[F][F];
float maxUP;

clock_t start, finish; 
double Z31,Z32,ngc1,ngc2,mf1,mf2,edft1,edft2;

#ifdef _Z3_
int release[F*K],deadline[F*K];
int sp[F*K];
int tC[SW][L],vC[SW][L][NUMPORT+1],cC[SW][L][Q];
Z3_solver s;
Z3_ast q[F],t[SW][L],v[SW][L][NUMPORT+1],c[SW][L][Q],alpha[F][K][N],beta[F][K][N];
Z3_ast lambda[SW][L];
Z3_ast ebar;
//Z3_context tctx;
Z3_ast assumptions[F+2*N*K*F];
int alphaC[F][K][N],betaC[F][K][N];
int spf[F*K],spk[F*K];
#endif

double Distance(int a,int b);
int GenTSNet();
int GenFlow();
int NoWaitingEDF();
int Dijkstra(int tlink[N][N],int s,int d,int *ph, int flid);
void AssignQ();
void AssignQFirstT(int * qAssg);
int Cap(int a,int b, int c, int d);
int Conflict(int g, int m,int s1,int a,int k,int s2);
int FexibleEDFWQ();
int AllFini(int * f);
int MyAlg();
int MoveForward(int f, int k, int g);
int UP();
int ORG();
void AssignNaive(int * q);
int NoWaitingAna();
int D(int f,int k, int x);
//int DFP(int f,int k, int x);
int Cap(int a,int b, int c, int d);
int Conflict(int g, int m,int s1,int a,int k,int s2);

#ifdef _Z3_
int SMTTSN();
int SMTTSNOpt();
int SMTTSNOptDev();
int Dev();
Z3_ast BigA(Z3_context ctx, int x,int k,int a,int b, int i,int j);
Z3_ast BigB(Z3_context ctx, int x,int i,int j,int a);
Z3_ast BigAOptDev(Z3_context ctx, int x,int k,int a,int b, int i,int j);
Z3_ast BigBOptDev(Z3_context ctx, int x,int i,int j,int a);
int Z3OptimizeSimple(int i);
int BeforebC(int f,int k);
int BeforeaC(int f,int k);
#endif

int _tmain(int argc, _TCHAR* argv[])
{
	char refilename[100];
	ofstream result;
	//int r;

	srand((unsigned) time (NULL));

	GenTSNet();
	GenFlow();
	
	REFILE;
	
	result.open(refilename,iostream::app);

	//analyze
	//if (UP()==1)
	//{
	//	result<<NoWaitingEDF()<<"\t"<<NoWaitingAna()<<endl;
	//}

	if (UP()==1)
	{
		//result<<Dev()<<"\t";
#ifdef _NWEDF_
		result<<NoWaitingEDF()<<"\t";
#endif
		//result<<MyAlg()<<"\t"<<FexibleEDFWQ()<<"\t1\t\t";  //nowaitingedf, my, UP
		result<<MyAlg()<<"\t1\t1\t\t"; 
		result<<MyEn<<"\t"<<ORG()<<"\t"<<FexEDFEn<<"\t\t";//my, org
		//result<<(int)(Z32-Z31)<<"\t";
		result<<(int)(ngc2-ngc1)<<"\t"<<(int)(mf2-mf1)<<"\t"<<(int)(edft2-edft1)<<"\t";
		result<<maxUP<<endl; //nowaiting,my
	}
	/*else
	{
		result<<"0\t0\t0\t0\t0\t\t";
		result<<"0\t0\t0\t0\t\t";
		result<<"0\t0\t0"<<endl;
	}*/
	
	result.close();
	//MyAlg();
#ifdef _FILEDELAY_
	ofstream outfile;
	outfile.open("delay.txt",iostream::out);

	int eF,eK;
	for (eF=0;eF<F;eF++)
	{
		for (eK=0;eK<(H/fl[eF].period);eK++)
		{
			//outfile<<delayNWEDF[eF][eK]<<"\t"<<delayNWEDFAna[eF][eK]<<"\t"<<delayNWEDFFex[eF][eK]<<"\t"<<delayNWEDFFexWQ[eF][eK]<<"\t"<<delayAna[eF][eK]<<endl;
			//outfile<<delayNWEDF[eF][eK]<<"\t"<<delayNWEDFFex[eF][eK]<<"\t"<<delayNWEDFFexWQ[eF][eK]<<"\t"<<delayFPEDF[eF][eK]<<"\t"<<delayMy[eF][eK]<<endl;//"\t"<<delayAna[eF][eK]<<endl;
			outfile<<delayNWEDF[eF][eK]<<"\t"<<delayNWEDFAna[eF][eK]<<endl;
		}
	}
	outfile<<endl;
	outfile.close();
#endif

//	FexibleEDFWQ();
//	MyAlg();
	return 0;
}

void AssignNaive(int * q)
{
	int eF,eF2;
	int sort[F],s;
	int a;
	
	for (eF=0;eF<F;eF++)
		sort[eF] = eF;

	for (eF=0;eF<(F-1);eF++)
	{
		for (eF2=eF+1;eF2<F;eF2++)
		{
			if (fl[sort[eF2]].deadline<fl[sort[eF]].deadline)
			{
				s = sort[eF];
				sort[eF] = sort[eF2];
				sort[eF2] = s;
			}
		}
	}

	a = F/Q;

	for (eF=0;eF<F;eF++)
	{
		q[sort[eF]] = eF/a;
	}
}

int ORG()
{
	int eF,eH,eN;
	int en[N]={0};

	ORGEn = 0;

	for (eF=0;eF<F;eF++)
	{
		for (eH=0;eH<fl[eF].c;eH++)
		{
			en[fl[eF].pathNode[eH]] += 2 * (H/fl[eF].period);
		}
	}

	for (eN=0;eN<SW;eN++)
	{
		if (ORGEn<en[eN])
			ORGEn = en[eN];
	}
	return ORGEn;
}

int UP()
{
	int eF,eH;
	float util[N][NUMPORT+1]={0.0};

	for (eF=0;eF<F;eF++)
		if (fl[eF].deadline<fl[eF].size*fl[eF].c)
			return 0;

	maxUP = 0;
	for (eF=0;eF<F;eF++)
	{
		for (eH=0;eH<fl[eF].c;eH++)
		{
			util[fl[eF].pathNode[eH]][fl[eF].pathPort[eH]] += (float)fl[eF].c/fl[eF].deadline;
			if (util[fl[eF].pathNode[eH]][fl[eF].pathPort[eH]]>1.0)
				return 0;
			if (maxUP<util[fl[eF].pathNode[eH]][fl[eF].pathPort[eH]])
				maxUP = util[fl[eF].pathNode[eH]][fl[eF].pathPort[eH]];
		}
	}
	//if ((max>(float)TH/10.0) && (max<(float)(TH+1.0)/10.0))
		return 1;
	//else
		//return 0;
}

int MyAlg()
{
	int eachN,eachF,eachH,eachS,eachP,eachK;
	int t;
	int state[F];
	//int cur[F];
	int dl[F],rl[F];
	int mindl,minF;
	//int s[N][NUMPORT+1][H]={0};
	int occupiedQ[N][NUMPORT+1][Q]={0};
//	int gateCon[N][NUMPORT+1][Q];
	int numofFiniPack[F];
	int r;

	AssignQFirstT(qC);
	//AssignNaive(qC);

	r = 1;
	for (eachN=0;eachN<N;eachN++)
		myEntries[eachN] = 1;

	for (eachN=0;eachN<N;eachN++)
		for (eachP=0;eachP<=NUMPORT+1;eachP++)
			for (t=0;t<H;t++)
				mys[eachN][eachP][t] = 0;
	
	for (eachF=0;eachF<F;eachF++)
	{
		numofFiniPack[eachF] = 0;
		state[eachF] = NULL;
		//cur[eachF] = 0;
		for (eachK=0;eachK<K;eachK++)
			delayMy[eachF][eachK] = MAX;
	}

	for (eachF=0;eachF<F;eachF++)
	{
		rl[eachF] = 0;
		dl[eachF] = fl[eachF].deadline;
	}

	mf1 = clock();

	while (!AllFini(numofFiniPack))
	{
		mindl = MAX;
		for (eachF=0;eachF<F;eachF++)
		{
			if (numofFiniPack[eachF]!=(H/fl[eachF].period))
			{
				if (dl[eachF]<mindl)
				{
					mindl = dl[eachF];
					minF = eachF;
				}
			}
		}
		if (mindl==MAX)
		{
			r = 0;
			break;
		}

		for (t=rl[minF];t<=(dl[minF]-fl[minF].size*fl[minF].c);t++)
		{
			for (eachH=0;eachH<fl[minF].c;eachH++)
			{
				for (eachS=0;eachS<fl[minF].size;eachS++)
				{
					if (mys[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+fl[minF].size*eachH+eachS]!=0)
						break;
				}
				if (eachS!=fl[minF].size)
					break;
			}
			if (eachH==fl[minF].c)
				break;
		}
		if (t<=(dl[minF]-fl[minF].size*fl[minF].c))
		{
			for (eachH=0;eachH<fl[minF].c;eachH++)
			{
				for (eachS=0;eachS<fl[minF].size;eachS++)
				{
					mys[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+fl[minF].size*eachH+eachS] = minF+1;
				}
			}
			delayMy[minF][t/fl[minF].period] = t+fl[minF].size*eachH - ((t/fl[minF].period)*fl[minF].period);
			numofFiniPack[minF] ++;
			rl[minF] = fl[minF].period * numofFiniPack[minF];
			dl[minF] = rl[minF] + fl[minF].deadline;
		}
		else
		{
			r = 0;
			delayMy[minF][t/fl[minF].period] = MAX;
			numofFiniPack[minF]++;
			rl[minF] = fl[minF].period * numofFiniPack[minF];
			dl[minF] = rl[minF] + fl[minF].deadline;
		}
	}

	//
	if (r==0)
	{
		r = 1;
		for (eachF=0;eachF<F;eachF++)
			numofFiniPack[eachF] = 0;

		for (eachF=0;eachF<F;eachF++)
		{
			rl[eachF] = 0;
			dl[eachF] = fl[eachF].deadline;
		}

		while (!AllFini(numofFiniPack))
		{
			mindl = MAX;
			for (eachF=0;eachF<F;eachF++)
			{
				if (numofFiniPack[eachF]!=(H/fl[eachF].period))
				{
					if (dl[eachF]<mindl)
					{
						mindl = dl[eachF];
						minF = eachF;
					}
				}
			}
			if (mindl==MAX)
			{
				r = 0;
				break;
			}
			
			delayMy[minF][numofFiniPack[minF]] = MoveForward(minF,numofFiniPack[minF],delayMy[minF][numofFiniPack[minF]]);
			numofFiniPack[minF] ++;
			rl[minF] = fl[minF].period * numofFiniPack[minF];
			dl[minF] = rl[minF] + fl[minF].deadline;
			if (delayMy[minF][numofFiniPack[minF]-1]==MAX)
			{
				r = 0;
				//break;
			}
		}
	}

	MyEn = 0;
	for (eachN=0;eachN<N;eachN++)                                                                                                  
	{
		if (myEntries[eachN]>MyEn)
			MyEn = myEntries[eachN];
	}

	mf2 = clock();
#ifdef _MY_
	ofstream outfile;
	outfile.open("MyAlgSch.txt",iostream::out);
	for (eachN=0;eachN<N;eachN++)
	{
		for (eachP=0;eachP<=NUMPORT;eachP++)
		{
			outfile<<eachN<<"."<<eachP<<"\t";
			for (t=0;t<H;t++)
			{
				if (mys[eachN][eachP][t]==0)
					outfile<<".\t";
				else
					outfile<<mys[eachN][eachP][t]<<"\t";
			}
			outfile<<endl;
		}
	}

	outfile<<endl;
	outfile.close();
#endif
	return r;
}

//return delay�� If delay is MAX, then cannot schedule. 
int MoveForward(int f, int k, int g) //g!=MAX, forward; g==MAX, schedule
{  //ÿ���ڵ�ӿ��أ�����queue
	int eS;
	int r,d,p;
	int t,t1,curH;

	r = k*fl[f].period;
	d = r + fl[f].deadline;

	if (g!=MAX)
	{
		curH = 0;
		t = r;
		p = 0;
		while (t<=(d-fl[f].size*fl[f].c+1))
		{
			if (mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t]==f+1)
				return (t+fl[f].size*fl[f].c);
			for (eS=0;eS<fl[f].size;eS++)
			{
				if ((mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t+eS]!=0) && (mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t+eS]!=f+1))
					break;
			}
			if (eS==fl[f].size)
			{
				if ((p!=(t-1)) && (p!=0))
					myEntries[fl[f].pathNode[curH]] += 2;
				for (t1=r;t1<=d;t1++)
					if (mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t1]== (f+1))
						mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t1] = 0;
				for (eS=0;eS<fl[f].size;eS++)
				{
					mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t+eS] = f + 1;
				}
				curH++;
				if (curH==fl[f].c)
					return (t+fl[f].size-r);
				t += eS;
				p = t - 1;
			}
			else
				t += (eS+1);
		}
	}
	else
	{
		p = 0;
		curH = 0;
		t = r;
		while (t<=(d-fl[f].size*(fl[f].c-curH)+1))
		{
			for (eS=0;eS<fl[f].size;eS++)
			{
				if (mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t+eS]!=0)
					break;
			}
			if (eS==fl[f].size)
			{
				if ((p!=(t-1)) && (p!=0)) 
					myEntries[fl[f].pathNode[curH]] += 2;
				for (eS=0;eS<fl[f].size;eS++)
				{
					mys[fl[f].pathNode[curH]][fl[f].pathPort[curH]][t+eS] = f + 1;
				}
				curH++;
				t --;
				p = t + eS;
				if (curH==fl[f].c)
					return (t+fl[f].size-r);
			}
			t += (eS+1);
		}
		return MAX;
	}
	return MAX;
}

int FexibleEDFWQ()
{
	int eachN,eachF,eachH,eachS,eachP,eachK,eachQ;
	int t;
	int state[F];
	int cur[F],startcur[F]={0};
	int dl[F];
	int mindl,minF;
	int s[N][NUMPORT+1][H]={0};
	int occupiedQ[N][NUMPORT+1][Q]={0};
	int numofentries[N]={0};
	int gateCon[N][NUMPORT+1][Q];

	AssignQFirstT(qC);
	//AssignNaive(qC);

	for (eachF=0;eachF<F;eachF++)
	{
		state[eachF] = NULL;
		cur[eachF] = startcur[eachF] = MAX;
		for (eachK=0;eachK<K;eachK++)
			delayNWEDFFexWQ[eachF][eachK] = MAX;
	}

	for (eachS=0;eachS<SW;eachS++)
	{
		numofentries[eachS] = 1;
		for (eachP=0;eachP<=NUMPORT;eachP++)
			for (eachQ=0;eachQ<Q;eachQ++)
				gateCon[eachS][eachP][eachQ] = 3;
	}

	edft1 = clock();
	for (t=0;t<H;t++)
	{
		for (eachF=0;eachF<F;eachF++)
		{
			if ((state[eachF]==ING)&&((t-startcur[eachF])==fl[eachF].size))
			{
				occupiedQ[fl[eachF].pathNode[cur[eachF]]][fl[eachF].pathPort[cur[eachF]]][qC[eachF]] = 0;
				state[eachF] = SRE;
				if (cur[eachF]>0)
					gateCon[fl[eachF].pathNode[cur[eachF]]][fl[eachF].pathPort[cur[eachF]]][qC[eachF]] = 2;
				cur[eachF] ++;
				if (cur[eachF]==fl[eachF].c)
				{
					occupiedQ[fl[eachF].pathNode[cur[eachF]]][fl[eachF].pathPort[cur[eachF]]][qC[eachF]] = 0;
					cur[eachF] = 0;
					state[eachF] = SFI;
					delayNWEDFFexWQ[eachF][(t-1)/fl[eachF].period] = t - ((t-1)/fl[eachF].period) * fl[eachF].period;
				}
			}
			if (t%fl[eachF].period==0)
			{
				if ((state[eachF]!=SFI) && (state[eachF]!=NULL))
				{
					delayNWEDFFexWQ[eachF][(t-1)/fl[eachF].period] = MAX;
					edft2 = clock();
					return 0;
				}
				state[eachF] = SRE;
				dl[eachF] = t + fl[eachF].deadline;
				cur[eachF] = 0;
			}
			else if (state[eachF]==SBL)
				state[eachF] = SRE;
			if ((t%fl[eachF].period>(fl[eachF].deadline-fl[eachF].size)) && (state[eachF]!=SFI) && (state[eachF]!=NULL))
			{
				delayNWEDFFexWQ[eachF][(t-1)/fl[eachF].period] = MAX;
				edft2 = clock();
				return 0;
			}
		}

		while(true)
		{
			mindl = MAX;
			for (eachF=0;eachF<F;eachF++)
			{
				if ((dl[eachF]>=t) && (state[eachF]==SRE))
				{
					if (dl[eachF]<mindl)
					{
						mindl = dl[eachF];
						minF = eachF;
					}
				}
			}
			if (mindl==MAX)
				break;

			eachH = cur[minF];
			{
				for (eachS=0;eachS<fl[minF].size;eachS++)
				{
					if ((s[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+eachS]!=0) || (occupiedQ[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][qC[minF]]==1))
					{
						if ((eachH>0) && (fl[minF].pathPort[eachH]!=NUMPORT))
						{
							if (gateCon[fl[minF].pathNode[eachH-1]][fl[minF].pathPort[eachH-1]][qC[minF]] != 0)
							{
								gateCon[fl[minF].pathNode[eachH-1]][fl[minF].pathPort[eachH-1]][qC[minF]] = 0;
								numofentries[fl[minF].pathNode[eachH-1]] ++;
							}
						}
						break;
					}
				}
			}
			if (eachS==fl[minF].size)
			{
				eachH = cur[minF];
				{
					for (eachS=0;eachS<fl[minF].size;eachS++)
					{
						s[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+eachS] = minF+1;
					}
				}
				state[minF] = ING;
				startcur[minF] = t;

				if (cur[minF]>0)
					occupiedQ[fl[minF].pathNode[cur[minF]-1]][fl[minF].pathPort[cur[minF]-1]][qC[minF]] = 0;
				if (cur[minF]!=fl[minF].c)
					occupiedQ[fl[minF].pathNode[cur[minF]]][fl[minF].pathPort[cur[minF]]][qC[minF]] = 1;
				if ((fl[minF].pathPort[cur[minF]-1]!=NUMPORT)  && (cur[minF]>1) && (gateCon[fl[minF].pathNode[cur[minF]-1]][fl[minF].pathPort[cur[minF]-1]][qC[minF]]==0))
				{
					gateCon[fl[minF].pathNode[cur[minF]-1]][fl[minF].pathPort[cur[minF]-1]][qC[minF]] = 3;
					numofentries[fl[minF].pathNode[cur[minF]-1]] ++;
				}
			}
			else
				state[minF] = SBL;
		}
	}

	FexEDFEn = 0;
	for (eachN=0;eachN<N;eachN++)
	{
		if (numofentries[eachN]>FexEDFEn)
			FexEDFEn = numofentries[eachN];
	}
	edft2 = clock();
#ifdef _EDFFEXWQ_
	ofstream outfile;
	outfile.open("EDFFexWQ.txt",iostream::out);
	for (eachN=0;eachN<N;eachN++)
	{
		for (eachP=0;eachP<=NUMPORT;eachP++)
		{
			outfile<<eachN<<"."<<eachP<<"\t";
			for (t=0;t<H;t++)
			{
				if (s[eachN][eachP][t]==0)
					outfile<<".\t";
				else
					outfile<<s[eachN][eachP][t]<<"\t";
			}
			outfile<<endl;
		}
	}

	for (eachN=0;eachN<SW;eachN++)
		outfile<<numofentries[eachN]<<endl;
	outfile<<endl;
	outfile.close();
#endif

	return 1;
}

int AllFini(int * f)
{
	int eF;
	for (eF=0;eF<F;eF++)
	{
		if (f[eF]!=(H/fl[eF].period))
			return 0;
	}
	return 1;
}


int Conflict(int g,int m,int s1,int a,int k,int s2)
{
	int gH,aH; //gS, aS
	int gt,at;
	int con;

	con = 0;

	gt = s1;
	at = s2;

	while ((gt<(fl[g].size*fl[g].c)) && (at<(fl[a].size*fl[a].c)))
	{
		aH = at/fl[a].size;
		gH = gt/fl[g].size;

		if ((fl[a].pathNode[aH]==fl[g].pathNode[gH]) &&(fl[a].pathNode[aH+1]==fl[g].pathNode[gH+1]))
			return 1;
		at ++;
		gt ++;
	}
	return 0;
}


int Cap(int a,int b, int c, int d)
{
	if ((c<=a)&&(a<=d))
		return 1;
	if ((a<=c)&&(c<=b))
		return 1;
	return 0;
}


int GenTSNet()
{
	int eachN,eachN2;
	int allcon;
	int s,e;

	nd[0].x = nd[0].y = EL/2;

	for (eachN=0;eachN<SW;eachN++)
		nd[eachN].type = 1;
	for (;eachN<N;eachN++)
		nd[eachN].type = 2;

again:	
	for (eachN=0;eachN<N;eachN++)
	{
		for (eachN2=0;eachN2<N;eachN2++)
			lk[eachN][eachN2].a = 0;
	}
	for (eachN=0;eachN<N;eachN++)
		nd[eachN].neighbor = nd[eachN].jointed = 0;

	nd[0].jointed = 1;
	
	allcon = 1;
	while (allcon)
	{
		allcon = 0;
		for (eachN=1;eachN<SW;eachN++)
		{
			if (nd[eachN].jointed == 0)
			{
				if(nd[eachN].neighbor==NUMPORT)
					goto again;
				nd[eachN].x = rand()%EL;
				nd[eachN].y = rand()%EL;
				allcon = 2;
			}
		}

		/*for (eachN=1;eachN<SW;eachN++)
		{
			if (nd[eachN].jointed == 0)
			{
				if(nd[eachN].neighbor==NUMPORT)
					goto again;
				allcon = 2;
			}
		}
		nd[1].x = 82; nd[1].y = 9;
		nd[2].x = 70; nd[2].y = 99;
		nd[3].x = 87; nd[3].y = 94;
		nd[4].x = 69; nd[4].y = 23;
		nd[5].x = 101; nd[5].y = 21;
		nd[6].x = 112; nd[6].y = 44;
		nd[7].x = 74; nd[7].y = 18;
		nd[8].x = 70; nd[8].y = 70;
		nd[9].x = 89; nd[9].y = 2;*/

		
		for (eachN=0;eachN<(SW-1);eachN++)
		{
			for (eachN2=eachN+1;eachN2<SW;eachN2++)
			{
				if ((lk[eachN][eachN2].a==0) && ((nd[eachN].jointed + nd[eachN2].jointed)>=1) && (Distance(eachN,eachN2)<CD))
				{
					if ((nd[eachN].neighbor<NUMPORT) && (nd[eachN2].neighbor<NUMPORT))
					{
						nd[eachN].jointed = nd[eachN2].jointed = 1;
						lk[eachN][eachN2].a = lk[eachN2][eachN].a = 1;
						lk[eachN][eachN2].sp = lk[eachN2][eachN].dp = nd[eachN].neighbor;
						lk[eachN2][eachN].sp = lk[eachN][eachN2].dp = nd[eachN2].neighbor;
						nd[eachN].neighbor ++;
						nd[eachN2].neighbor ++;
						allcon = 1;
					}
				}
			}
		}
		if (allcon==2)
			goto again;

	}

	for (eachN=0;eachN<SW;eachN++)
	{
		for (eachN2=0;eachN2<ES;eachN2++)
		{
			e = SW + eachN*ES+eachN2;
			s = eachN;
			lk[e][s].a = lk[s][e].a = 1;
			lk[e][s].sp = lk[s][e].dp = 0;
			lk[s][e].sp = lk[e][s].dp = NUMPORT;
		}
	}

#ifdef _FILENET_
	ofstream outfile;
	outfile.open("net.txt",iostream::out);
	for (eachN=0;eachN<N;eachN++)
	{
		outfile<<"nd["<<eachN<<"] ("<<nd[eachN].x<<", "<<nd[eachN].y<<"): ";
		if (nd[eachN].type==1)
			outfile<<"SW"<<endl;
		else
			outfile<<"ES"<<endl;
	}

	for (eachN=0;eachN<N;eachN++)
	{
		for (eachN2=0;eachN2<N;eachN2++)
		{
			if (lk[eachN][eachN2].a==1)
			{
				outfile<<"link "<<eachN<<"."<< lk[eachN][eachN2].sp<<" - "<<eachN2<<"."<<lk[eachN][eachN2].dp<<endl;
			}
		}
	}
	outfile<<endl;
	outfile.close();
#endif
	return 1;
}

int GenFlow()
{
	int eachF;
	int eachN,eachN2,eachH;
	int tlink[N][N];

	for (eachN=0;eachN<N;eachN++)
	{
		for (eachN2=0;eachN2<N;eachN2++)
		{
			tlink[eachN][eachN2] = (lk[eachN][eachN2].a)?1:MAX;
		}
	}

	for (eachF=0;eachF<F;eachF++)
	{
		fl[eachF].s = SW + rand()%SW;
		fl[eachF].d = SW + rand()%SW;
		while (fl[eachF].d == fl[eachF].s)
			fl[eachF].d = SW + rand()%SW;
		fl[eachF].c = Dijkstra(tlink,fl[eachF].s,fl[eachF].d,fl[eachF].pathNode,eachF);
		for (eachH=0;eachH<fl[eachF].c;eachH++)
		{
			tlink[fl[eachF].pathNode[eachH]][fl[eachF].pathNode[eachH+1]] ++;
			tlink[fl[eachF].pathNode[eachH+1]][fl[eachF].pathNode[eachH]] ++;

			//if (eachH!=fl[eachF].c-1)
				fl[eachF].pathPort[eachH] = lk[fl[eachF].pathNode[eachH]][fl[eachF].pathNode[eachH+1]].sp;
		}

		fl[eachF].period = pow2[(rand()%(PEUP-PELOW+1) + PELOW)] * PEUNIT;
		fl[eachF].size = rand()%(SIZEUP-SIZELOW+1) + SIZELOW;
		fl[eachF].deadline = (int)((fl[eachF].period - fl[eachF].size) * ((float)(rand()%50+50)/100) + fl[eachF].size);
	}

	

#ifdef _FILEFLOW_
	ofstream outfile;
	outfile.open("flows.txt",iostream::out);
	for (eachF=0;eachF<F;eachF++)
	{
		outfile<<eachF<<" from "<<fl[eachF].s<<" to "<<fl[eachF].d<<", num of hops "<<fl[eachF].c<<", period "<<fl[eachF].period<<", size "<<fl[eachF].size<<", deadline "<<fl[eachF].deadline<<endl;
		for (eachN=0;eachN<=fl[eachF].c;eachN++)
		{
			outfile<<"("<<eachN<<") "<<fl[eachF].pathNode[eachN]<<" "<<fl[eachF].pathPort[eachN];
		}
		outfile<<endl;
	}
	outfile<<endl;
	outfile.close();
#endif
	return 1;
}

int NoWaitingEDF()
{
	int eachF,eachH,eachS,eachK;
	int t;
	int state[F];
	int cur[F];
	int dl[F];
	int mindl,minF;
	int s[N][NUMPORT+1][H]={0};

	for (eachF=0;eachF<F;eachF++)
	{
		state[eachF] = NULL;
		cur[eachF] = 0;
		for (eachK=0;eachK<K;eachK++)
			delayNWEDF[eachF][eachK] = MAX;
	}

	ngc1 = clock();
	for (t=0;t<H;t++)
	{
		for (eachF=0;eachF<F;eachF++)
		{
			if (t%fl[eachF].period==0)
			{
				if ((state[eachF]!=SFI) && (state[eachF]!=NULL))
				{
					ngc2 = clock();
					return 0;
				}
				state[eachF] = SRE;
				dl[eachF] = t + fl[eachF].deadline;
				cur[eachF] = 0;
			}
			else if (state[eachF]==SBL)
				state[eachF] = SRE;
			if ((t%fl[eachF].period>(fl[eachF].deadline-fl[eachF].size * fl[eachF].c)) && (state[eachF]!=SFI) && (state[eachF]!=NULL))
			{
					ngc2 = clock();
					return 0;
			}
		}

		while(true)
		{
			mindl = MAX;
			for (eachF=0;eachF<F;eachF++)
			{
				if ((dl[eachF]>=t) && (state[eachF]==SRE))
				{
					if (dl[eachF]<mindl)
					{
						mindl = dl[eachF];
						minF = eachF;
					}
				}
			}
			if (mindl==MAX)
				break;

			//sch
		
			for (eachH=0;eachH<fl[minF].c;eachH++)
			{
				for (eachS=0;eachS<fl[minF].size;eachS++)
				{
					if (s[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+fl[minF].size*eachH+eachS]!=0)
						break;
				}
				if (eachS!=fl[minF].size)
					break;
			}
			if (eachH==fl[minF].c)
			{
				for (eachH=0;eachH<fl[minF].c;eachH++)
				{
					for (eachS=0;eachS<fl[minF].size;eachS++)
					{
						s[fl[minF].pathNode[eachH]][fl[minF].pathPort[eachH]][t+fl[minF].size*eachH+eachS] = minF+1;
					}
				}
				delayNWEDF[minF][t/fl[minF].period] = t+fl[minF].size*eachH - ((t/fl[minF].period)*fl[minF].period);
				state[minF] = SFI;
				//cur[minF] ++;
				//if (cur[minF]==fl[minF].c)
					//state[minF] = SFI;
			}
			else
				state[minF] = SBL;
		}

	}
#ifdef _SCHORIGINAL_
	ofstream outfile;
	outfile.open("NWEDF.txt",iostream::out);
	for (eachN=0;eachN<N;eachN++)
	{
		for (eachP=0;eachP<=NUMPORT;eachP++)
		{
			outfile<<eachN<<"."<<eachP<<"\t";
			for (t=0;t<H;t++)
			{
				if (s[eachN][eachP][t]==0)
					outfile<<".\t";
				else
					outfile<<s[eachN][eachP][t]<<"\t";
			}
			outfile<<endl;
		}
	}

	outfile<<endl;
	outfile.close();
#endif
	ngc2 = clock();
	return 1;
}



double Distance(int a,int b)
{
	double i,j;
	i = abs(nd[a].x - nd[b].x);
	j = abs(nd[a].y - nd[b].y);
	return sqrt(i*i+j*j);
}

int Dijkstra(int tlink[N][N],int s,int d,int *ph, int flid)
{
	int i,j,k;
	int mark[N];
//	int path[N];
	int dis[N];
	int allmark;
	int min;
	int pre[N];
	int temp[N];
	int path[N];
	for (i=0;i<N;i++)
	{
		mark[i] = MAX;
		path[i] = N;
		if (i!=s)
		{
			dis[i] = (tlink[s][i]==0)?MAX:tlink[s][i];
			pre[i] = s;
		}
		else
		{
			dis[i] = 0;
			pre[i] = N;
		}
	}
	mark[s] = 1;
	allmark = 1;
	while(allmark!=N)
	{
		min = MAX;
		for (i=0;i<N;i++)
		{
			if ((mark[i]==MAX) && (dis[i]<min))
			{
				min = dis[i];
				k = i;
			}
		}
		mark[k] = 1;
		allmark++;
		for (i=0;i<N;i++)
		{
			if (mark[i]==MAX && dis[i]>(dis[k]+tlink[k][i]))
			{
				dis[i] = dis[k] + tlink[k][i];
				pre[i] = k;
			}
		}
	}
	i = pre[d];
	j = 0;
	temp[j] = i;
	j++;
	while (i!=s)
	{
		i = pre[i];
		temp[j] = i;
		j++;
	}
	for (i=0;i<j;i++)
	{
		path[i] = temp[j-i-1];
	}
	path[j] = d;

	//if (flag==1)
	//{
	//	i = fl[flid].c;
	//	j +=fl[flid].c; 
	//}
	//else
		i = 0;
	k = 0;
	while (i<N)
	{
		ph[i] = path[k];
		i ++;
		k ++;
	}
	return j;
}



void AssignQ()
{
	//int qAssg[F];
	//num of queues: Q
	
	int oh[F];
	int eachF,eachF2,eachH,eachS,eachP,eachQ;
	int sort[F];
	int aq[F];
	int max;
	int maxf,swvar;
	int maxoh[Q];
	int i;
	int minQ;
	int minQva;
	int t;

	for (eachH=0;eachH<H;eachH++)
		for (eachS=0;eachS<SW;eachS++)
			for (eachP=0;eachP<=NUMPORT;eachP++)
				for (eachQ=0;eachQ<Q;eachQ++)
					curwl[eachH][eachS][eachP][eachQ] = 0;

	for (eachF=0;eachF<F;eachF++)
		sort[eachF] = eachF;

	for (eachF=0;eachF<F;eachF++)
	{
		oh[eachF] = (int)((float)fl[eachF].c/(float)fl[eachF].deadline*10000.0);
#ifdef _AQDEBUG_
		cout<<"oh["<<eachF<<"] "<<oh[eachF]<<endl;
#endif
	}

	for (eachF=0;eachF<(F-1);eachF++)
	{
		max = oh[sort[eachF]];
		for (eachF2=eachF+1;eachF2<F;eachF2++)
		{
			if (max<oh[sort[eachF2]])
			{
				max = oh[sort[eachF2]];
				maxf = eachF2;
			}
		}
		if (max!=oh[eachF])
		{
			swvar = sort[eachF];
			sort[eachF] = sort[maxf];
			sort[maxf] = swvar;
		}
	}

#ifdef _AQDEBUG_
	for (eachF=0;eachF<F;eachF++)
	{
		cout<<"sort["<<eachF<<"] "<<sort[eachF]<<endl;
	}
#endif

	for (eachF=0;eachF<F;eachF++)
	{
		eachF2 = sort[eachF];
#ifdef _AQDEBUG_
		cout<<"---------\nthe flow "<<eachF2<<endl;
#endif
		for (eachQ=0;eachQ<Q;eachQ++)
		{
			maxoh[eachQ] = 0;
			for (t=0;t<H;t++)
			{
				if ((t%fl[eachF2].period)<fl[eachF2].deadline)
				{
					for (i=1;i<fl[eachF2].c;i++)
					{
						if (maxoh[eachQ]<curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][eachQ])
						{
							maxoh[eachQ] = curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][eachQ];
						}
					}
				}
			}
#ifdef _AQDEBUG_
			cout<<"maxoh["<<eachQ<<"] "<<maxoh[eachQ]<<endl;
#endif
		}


		minQva = MAX;
		for (eachQ=0;eachQ<Q;eachQ++)
		{
			if (minQva>maxoh[eachQ])
			{
				minQva = maxoh[eachQ];
				minQ = eachQ;
			}
		}
		aq[eachF2] = minQ;
#ifdef _AQDEBUG_
		cout<<"assign the queue "<<minQ<<" to the flow "<<eachF2<<endl;
#endif
		for (t=0;t<H;t++)
		{
			if ((t%fl[eachF2].period)<fl[eachF2].deadline)
			{
				for (i=1;i<fl[eachF2].c;i++)
				{
					curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][minQ] += oh[eachF2];
				}
			}
		}
	}

#ifdef _AQDEBUG_
	for (eachF=0;eachF<F;eachF++)
	{
		cout<<eachF<<" "<<aq[eachF]<<endl;
	}
	cout<<endl;
#endif
}

void AssignQFirstT(int * qAssg)
{
	//int qAssg[F];
	//num of queues: Q
	
	int oh[F];
	int eachF,eachF2,eachH,eachS,eachP,eachQ;
	int sort[F];
	//int aq[F];
	int * aq;
	int max;
	int maxf,swvar;
	int maxoh[Q];
	int i;
	int minQ;
	int minQva;
	int t;

	aq = qAssg;

	for (eachH=0;eachH<H;eachH++)
		for (eachS=0;eachS<SW;eachS++)
			for (eachP=0;eachP<=NUMPORT;eachP++)
				for (eachQ=0;eachQ<Q;eachQ++)
					curwl[eachH][eachS][eachP][eachQ] = 0;

	for (eachF=0;eachF<F;eachF++)
		sort[eachF] = eachF;

	for (eachF=0;eachF<F;eachF++)
	{
		oh[eachF] = (int)((float)fl[eachF].c/(float)fl[eachF].deadline*10000.0);
#ifdef _AQDEBUG_
		cout<<"oh["<<eachF<<"] "<<oh[eachF]<<endl;
#endif
	}

	for (eachF=0;eachF<(F-1);eachF++)
	{
		max = oh[sort[eachF]];
		for (eachF2=eachF+1;eachF2<F;eachF2++)
		{
			if (max<oh[sort[eachF2]])
			{
				max = oh[sort[eachF2]];
				maxf = eachF2;
			}
		}
		if (max!=oh[eachF])
		{
			swvar = sort[eachF];
			sort[eachF] = sort[maxf];
			sort[maxf] = swvar;
		}
	}

#ifdef _AQDEBUG_
	for (eachF=0;eachF<F;eachF++)
	{
		cout<<"sort["<<eachF<<"] "<<sort[eachF]<<endl;
	}
#endif

	for (eachF=0;eachF<F;eachF++)
	{
		eachF2 = sort[eachF];
#ifdef _AQDEBUG_
		cout<<"---------\nthe flow "<<eachF2<<endl;
#endif
		for (eachQ=0;eachQ<Q;eachQ++)
		{
			maxoh[eachQ] = 0;
			for (t=0;t<1;t++)
			{
				//if ((t%fl[eachF2].period)<fl[eachF2].deadline)
				{
					for (i=1;i<fl[eachF2].c;i++)
					{
						if (maxoh[eachQ]<curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][eachQ])
						{
							maxoh[eachQ] = curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][eachQ];
						}
					}
				}
			}
#ifdef _AQDEBUG_
			cout<<"maxoh["<<eachQ<<"] "<<maxoh[eachQ]<<endl;
#endif
		}


		minQva = MAX;
		for (eachQ=0;eachQ<Q;eachQ++)
		{
			if (minQva>maxoh[eachQ])
			{
				minQva = maxoh[eachQ];
				minQ = eachQ;
			}
		}
		aq[eachF2] = minQ;
#ifdef _AQDEBUG_
		cout<<"assign the queue "<<minQ<<" to the flow "<<eachF2<<endl;
#endif
		for (t=0;t<1;t++)
		{
			//if ((t%fl[eachF2].period)<fl[eachF2].deadline)
			{
				for (i=1;i<fl[eachF2].c;i++)
				{
					curwl[t][fl[eachF2].pathNode[i]][fl[eachF2].pathPort[i]][minQ] += oh[eachF2];
				}
			}
		}
	}

#ifdef _AQDEBUG_
	for (eachF=0;eachF<F;eachF++)
	{
		cout<<eachF<<" "<<aq[eachF]<<endl;
	}
	cout<<endl;
#endif
}

#ifdef _Z3_

int BeforebC(int f,int k)
{
	int eachP;
	for (eachP=0;eachP<=bC;eachP++)
		if ((spf[sp[eachP]]==f)&&(spk[sp[eachP]]==k))
			return 1;
	return 0;
}

int BeforeaC(int f,int k)
{
	int eachP;
	for (eachP=0;eachP<aC;eachP++)
		if ((spf[sp[eachP]]==f)&&(spk[sp[eachP]]==k))
			return 1;
	return 0;
}
int Z3OptimizeSimple(int i)
{
	char str[100];
	Z3_ast one[2];
	Z3_ast result;
	Z3_model m=0;
	Z3_lbool r;
	Z3_context tctx;
	Z3_optimize opt;
	tctx     = mk_context();
	opt = Z3_mk_optimize(tctx);

	/*one[0] = mk_int(ctx,1);
	one[1] = mk_int(ctx,1);*/
	sprintf_s (str,"result");
	result = mk_int_var(tctx,str);

	Z3_optimize_assert(tctx,opt,Z3_mk_le(tctx,mk_int(tctx,1),result));
	Z3_optimize_assert(tctx,opt,Z3_mk_le(tctx,result,mk_int(tctx,i)));

	Z3_optimize_maximize(tctx,opt,result);

	r = Z3_optimize_check(tctx,opt,0,one);
	switch(r)
	{
	case Z3_L_TRUE:
		m= Z3_optimize_get_model(tctx,opt);
		if (m) Z3_model_inc_ref(tctx,m);
		display_model(tctx,stdout,m);
		break;
	default:
		printf("err");
	}

	Z3_optimize_dec_ref(tctx, opt);
	Z3_del_context(tctx);
    
	return 1;
}

int SMTTSN()
{
	char str[100];
	int eachF,eachN,eachSW,eachL,eachP,eachQ,eachK;
	int z,zi,b,a,g,h,m,k,i,j,y,yi;
	Z3_ast e1,e2;
	Z3_ast arg1[MAXA],arg2[MAXA],arg3[MAXA],arg4[MAXA],arg5[MAXA],arg6[Q];
	Z3_ast one,zero;
	Z3_context ctx;

#ifdef _LOG_Z3_CALLS_
    Z3_open_log("z3.log");
    LOG_MSG("SMT TSN");
#endif

	//printf("\nfind_model_example1\n");

	ctx     = mk_context();
    s       = mk_solver(ctx);

	one = mk_int(ctx,1);
	zero = mk_int(ctx,0);

	//q[F]
	for (eachF=0;eachF<F;eachF++)
	{
		sprintf_s (str,"q_%d",eachF);
		q[eachF] = mk_int_var(ctx,str);
	}

	//t[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"t_%d_%d",eachSW,eachL);
			t[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	//v[SW][L][NUMPORT]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				sprintf_s (str,"v_%d_%d_%d",eachSW,eachL,eachP);
				v[eachSW][eachL][eachP] = mk_int_var(ctx,str);
			}
		}
	}

	//c[SW][L][Q]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				sprintf_s (str,"c_%d_%d_%d",eachSW,eachL,eachQ);
				c[eachSW][eachL][eachQ] = mk_int_var(ctx,str);
			}
		}
	}

	//alpha[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"alpha_%d_%d_%d",eachF,eachK,eachN);
				alpha[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	//beta[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"beta_%d_%d_%d",eachF,eachK,eachN);
				beta[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	// 1)
	//cout<<"1";
	for (eachSW=0;eachSW<SW;eachSW++)
		Z3_solver_assert(ctx,s,Z3_mk_eq(ctx,t[eachSW][0],one));
	
	//cout<<"2";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=1;eachL<L;eachL++)
		{
			zi = 0;
			for (z=(eachL+1);z<L;z++)
			{
				arg1[zi] = Z3_mk_eq(ctx,t[eachSW][z],zero);
				zi ++;
			}
			arg2[0] = Z3_mk_and(ctx,zi,arg1);
			arg2[1] = Z3_mk_eq(ctx,t[eachSW][eachL],zero);
			arg4[0] = Z3_mk_and (ctx,2,arg2);

			arg3[0] = Z3_mk_le(ctx,one,t[eachSW][eachL-1]);
			arg3[1] = Z3_mk_lt(ctx,t[eachSW][eachL-1],t[eachSW][eachL]);
			arg3[2] = Z3_mk_le(ctx,t[eachSW][eachL],mk_int(ctx,H));
			arg3[3] = Z3_mk_gt(ctx,t[eachSW][eachL],zero);
			arg4[1] = Z3_mk_and (ctx,4,arg3);

			e1 = Z3_mk_or (ctx,2,arg4);

			Z3_solver_assert(ctx,s,e1);
		}
	}
	
	//cout<<"3";

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				arg1[0] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],mk_int(ctx,(eachQ+1)));
				arg1[1] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],zero);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	
	//cout<<"4";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				arg1[0] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],zero);
				arg1[1] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],one);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	//cout<<"5";

	for (eachF=0;eachF<F;eachF++)
	{
		arg1[0] = Z3_mk_le(ctx,one,q[eachF]);
		arg1[1] = Z3_mk_le(ctx,q[eachF],mk_int(ctx,Q));
		e1 = Z3_mk_and(ctx,2,arg1);
		Z3_solver_assert(ctx,s,e1);
	}
	//cout<<"6";

	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			for (b=0;b<fl[eachF].c;b++)
			{
				arg1[0] = Z3_mk_le(ctx,mk_int(ctx,eachK*fl[eachF].period),alpha[eachF][eachK][b]);
				arg1[1] = Z3_mk_le(ctx,alpha[eachF][eachK][b],beta[eachF][eachK][b]);
				arg1[2] = Z3_mk_le(ctx,beta[eachF][eachK][b],mk_int(ctx,eachK*fl[eachF].period+fl[eachF].deadline-1));
				e1 = Z3_mk_and(ctx,3,arg1);
				Z3_solver_assert(ctx,s,e1);
				
				arg2[0] = alpha[eachF][eachK][b];
				arg2[1] = mk_int(ctx,fl[eachF].size-1);
				e1 = Z3_mk_eq(ctx,beta[eachF][eachK][b],Z3_mk_add(ctx,2,arg2));
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	//cout<<"7";

	// 2)
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			for (b=0;b<(fl[eachF].c-1);b++)
			{
				e1 = Z3_mk_lt(ctx,beta[eachF][eachK][b],alpha[eachF][eachK][b+1]);
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	//cout<<"8";

	// 3) => 1)

	// 4) .....  change symbols to follow those in mu paper
	for (a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						for (b=0;b<fl[a].c;b++)
						{
							for (h=0;h<fl[g].c;h++)
							{
								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
								e1 = Z3_mk_and(ctx,2,arg1);
								arg2[0] = Z3_mk_not(ctx,e1);

								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b+1]),mk_int(ctx,fl[g].pathNode[h+1]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b+1]),mk_int(ctx,fl[g].pathPort[h+1]));
								e2 = Z3_mk_and(ctx,2,arg1);
								arg2[1] = Z3_mk_not(ctx,e2);

								arg2[2] = Z3_mk_gt(ctx,alpha[a][k][b],beta[g][m][h]);
								arg2[3] = Z3_mk_gt(ctx,alpha[g][m][h],beta[a][k][b]);

								e1 = Z3_mk_or(ctx,4,arg2);

								Z3_solver_assert(ctx,s,e1);
							}
						}
					}
				}
			}
		}
	}
	//cout<<"9";

	// 5) ......
	for(a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						for (b=1;b<fl[a].c;b++)
						{
							for (h=1;h<fl[g].c;h++)
							{
								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
								e1 = Z3_mk_and(ctx,2,arg1);
								arg2[0] = Z3_mk_not(ctx,e1);

								arg2[1] = Z3_mk_not(ctx,Z3_mk_eq(ctx,q[a],q[g]));

								arg2[2] = Z3_mk_lt(ctx,beta[a][k][b],alpha[g][m][h-1]);
								arg2[3] = Z3_mk_lt(ctx,beta[g][m][h],alpha[a][k][b-1]);

								e1 = Z3_mk_or(ctx,4,arg2);

								Z3_solver_assert(ctx,s,e1);
							}
						}
					}
				}
			}
		}
	}
	//cout<<"A";

	// 6)			
	for (a=0;a<F;a++)
	{
		for (b=1;b<fl[a].c;b++)
		{
			//fl[a].pathNode[b]; fl[a].pathPort[b];
			for (k=0;k<(H/fl[a].period);k++)
			{
				zi = 0;
				for (g=k*fl[a].period;g<(k*fl[a].period+fl[a].deadline);g++)
				{
					arg2[0] = Z3_mk_eq(ctx,alpha[a][k][b],mk_int(ctx,g));
					arg2[1] = BigA(ctx,g,k,a,b,fl[a].pathNode[b],fl[a].pathPort[b]);
					arg1[zi] = Z3_mk_and(ctx,2,arg2);
					zi ++;
				}
				e1 = Z3_mk_or(ctx,zi,arg1);
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	//cout<<"B";

	// 7)
	for (a=0;a<F;a++)
	{
		for (k=0;k<(H/fl[a].period);k++)
		{
			for (b=1;b<fl[a].c;b++)
			{
				i = fl[a].pathNode[b];
				j = fl[a].pathPort[b];
				yi = 0;
				for (y=0;y<L;y++)
				{
					for(eachQ=0;eachQ<Q;eachQ++)
					{
						arg6[eachQ] = Z3_mk_eq(ctx,c[i][y][eachQ],q[a]);
					}
					arg4[1] = Z3_mk_or(ctx,Q,arg6);
					arg5[0] = Z3_mk_le(ctx,alpha[a][k][b],t[i][y]);
					arg5[1] = Z3_mk_le(ctx,t[i][y],beta[a][k][b]);
					arg4[0] = Z3_mk_and(ctx,2,arg5);
					arg3[2] = Z3_mk_and(ctx,2,arg4);
					arg3[0] = Z3_mk_lt(ctx,t[i][y],alpha[a][k][b]);
					arg3[1] = Z3_mk_lt(ctx,beta[a][k][b],t[i][y]);
					arg2[2] = Z3_mk_or(ctx,3,arg3); 
					arg2[0] = Z3_mk_eq(ctx,t[i][y],zero);
					arg2[1] = Z3_mk_eq(ctx,v[i][y][j],zero);		
					arg1[yi] = Z3_mk_or(ctx,3,arg2);
					yi ++;
				}

				e1 = Z3_mk_and(ctx,yi,arg1);
				//e1 = Z3_mk_eq(ctx,t[i][y],zero);
				Z3_solver_assert(ctx,s,e1);
			}
		}
	}
	//cout<<"C";

	//lambda[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"lambda_%d_%d",eachSW,eachL);
			lambda[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		yi = 0;
		for (y=0;y<L;y++)
		{
			arg1[0] = Z3_mk_eq(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],zero);
			arg2[0] = Z3_mk_and(ctx,2,arg1);

			arg1[0] = Z3_mk_gt(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],one);
			arg2[1] = Z3_mk_and(ctx,2,arg1);

			arg3[yi] = Z3_mk_or(ctx,2,arg2);
			yi ++;
		}
		e1 = Z3_mk_and(ctx,yi,arg3);
		Z3_solver_assert(ctx,s,e1);
	}

	sprintf_s (str,"ebar");
	ebar = mk_int_var(ctx,str);

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (y=0;y<L;y++)
		{
			arg1[y] = lambda[eachSW][y];
		}
		e1 = Z3_mk_add(ctx,y,arg1);
		Z3_solver_assert(ctx,s,Z3_mk_le(ctx,e1,ebar));
	}


    //Z3_solver_assert(ctx, s, x_xor_y);

    printf("Z3 is running...\n");
    check(ctx, s, Z3_L_TRUE);

    del_solver(ctx, s);
    Z3_del_context(ctx);

#ifdef _LOG_Z3_CALLS_
    Z3_close_log();
#endif

	return 1;
}

int SMTTSNOpt()
{
	char str[100];
	Z3_optimize opt;
	int eachF,eachN,eachSW,eachL,eachP,eachQ,eachK;
	int z,zi,b,a,g,h,m,k,i,j,y,yi;
	Z3_ast e1,e2;
	Z3_ast arg1[MAXA],arg2[MAXA],arg3[MAXA],arg4[MAXA],arg5[MAXA],arg6[Q];
	Z3_ast one,zero;
	Z3_model mr=0;
	Z3_lbool r;
	Z3_context ctx;
	//Z3_optimize opt;

#ifdef _LOG_Z3_CALLS_
    Z3_open_log("z3.log");
    LOG_MSG("SMT TSN Opt");
#endif

	//printf("\nfind_model_example1\n");

	ctx     = mk_context();
    //s       = mk_solver(ctx);
	opt = Z3_mk_optimize(ctx);

	one = mk_int(ctx,1);
	zero = mk_int(ctx,0);

	//q[F]
	for (eachF=0;eachF<F;eachF++)
	{
		sprintf_s (str,"q_%d",eachF);
		q[eachF] = mk_int_var(ctx,str);
	}

	//t[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"t_%d_%d",eachSW,eachL);
			t[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	//v[SW][L][NUMPORT]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				sprintf_s (str,"v_%d_%d_%d",eachSW,eachL,eachP);
				v[eachSW][eachL][eachP] = mk_int_var(ctx,str);
			}
		}
	}

	//c[SW][L][Q]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				sprintf_s (str,"c_%d_%d_%d",eachSW,eachL,eachQ);
				c[eachSW][eachL][eachQ] = mk_int_var(ctx,str);
			}
		}
	}

	//alpha[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"alpha_%d_%d_%d",eachF,eachK,eachN);
				alpha[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	//beta[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"beta_%d_%d_%d",eachF,eachK,eachN);
				beta[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	// 1)
	//cout<<"1";
	for (eachSW=0;eachSW<SW;eachSW++)
		Z3_optimize_assert(ctx,opt,Z3_mk_eq(ctx,t[eachSW][0],one));
	
	//cout<<"2";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=1;eachL<L;eachL++)
		{
			// zi used to count the number of arg1
			// z indicates all following entries after eachL
			zi = 0;
			for (z=(eachL+1);z<L;z++)
			{
				arg1[zi] = Z3_mk_eq(ctx,t[eachSW][z],zero);
				zi ++;
			}
			arg2[0] = Z3_mk_and(ctx,zi,arg1);
			arg2[1] = Z3_mk_eq(ctx,t[eachSW][eachL],zero);
			arg4[0] = Z3_mk_and (ctx,2,arg2);

			arg3[0] = Z3_mk_le(ctx,one,t[eachSW][eachL-1]);
			arg3[1] = Z3_mk_lt(ctx,t[eachSW][eachL-1],t[eachSW][eachL]);
			arg3[2] = Z3_mk_le(ctx,t[eachSW][eachL],mk_int(ctx,H));
			arg3[3] = Z3_mk_gt(ctx,t[eachSW][eachL],zero);
			arg4[1] = Z3_mk_and (ctx,4,arg3);

			e1 = Z3_mk_or (ctx,2,arg4);

			Z3_optimize_assert(ctx,opt,e1);
		}
	}
	
	//cout<<"3";

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				arg1[0] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],mk_int(ctx,(eachQ+1)));
				arg1[1] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],zero);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	
	//cout<<"4";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				arg1[0] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],zero);
				arg1[1] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],one);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"5";

	for (eachF=0;eachF<F;eachF++)
	{
		arg1[0] = Z3_mk_le(ctx,one,q[eachF]);
		arg1[1] = Z3_mk_le(ctx,q[eachF],mk_int(ctx,Q));
		e1 = Z3_mk_and(ctx,2,arg1);
		Z3_optimize_assert(ctx,opt,e1);
	}
	//cout<<"6";

	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			for (b=0;b<fl[eachF].c;b++)
			{
				arg1[0] = Z3_mk_le(ctx,mk_int(ctx,eachK*fl[eachF].period),alpha[eachF][eachK][b]);
				arg1[1] = Z3_mk_le(ctx,alpha[eachF][eachK][b],beta[eachF][eachK][b]);
				arg1[2] = Z3_mk_le(ctx,beta[eachF][eachK][b],mk_int(ctx,eachK*fl[eachF].period+fl[eachF].deadline-1));
				e1 = Z3_mk_and(ctx,3,arg1);
				Z3_optimize_assert(ctx,opt,e1);
				
				arg2[0] = alpha[eachF][eachK][b];
				arg2[1] = mk_int(ctx,fl[eachF].size-1);
				e1 = Z3_mk_eq(ctx,beta[eachF][eachK][b],Z3_mk_add(ctx,2,arg2));
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"7";

	// 2)
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			for (b=0;b<(fl[eachF].c-1);b++)
			{
				e1 = Z3_mk_lt(ctx,beta[eachF][eachK][b],alpha[eachF][eachK][b+1]);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"8";

	// 3) => 1)

	// 4) .....  change symbols to follow those in mu paper
	for (a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						for (b=0;b<fl[a].c;b++)
						{
							for (h=0;h<fl[g].c;h++)
							{
								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
								e1 = Z3_mk_and(ctx,2,arg1);
								arg2[0] = Z3_mk_not(ctx,e1);

								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b+1]),mk_int(ctx,fl[g].pathNode[h+1]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b+1]),mk_int(ctx,fl[g].pathPort[h+1]));
								e2 = Z3_mk_and(ctx,2,arg1);
								arg2[1] = Z3_mk_not(ctx,e2);

								arg2[2] = Z3_mk_gt(ctx,alpha[a][k][b],beta[g][m][h]);
								arg2[3] = Z3_mk_gt(ctx,alpha[g][m][h],beta[a][k][b]);

								e1 = Z3_mk_or(ctx,4,arg2);

								Z3_optimize_assert(ctx,opt,e1);
							}
						}
					}
				}
			}
		}
	}
	//cout<<"9";

	// 5) ......
	for(a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						for (b=1;b<fl[a].c;b++)
						{
							for (h=1;h<fl[g].c;h++)
							{
								arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
								arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
								e1 = Z3_mk_and(ctx,2,arg1);
								arg2[0] = Z3_mk_not(ctx,e1);

								arg2[1] = Z3_mk_not(ctx,Z3_mk_eq(ctx,q[a],q[g]));

								arg2[2] = Z3_mk_lt(ctx,beta[a][k][b],alpha[g][m][h-1]);
								arg2[3] = Z3_mk_lt(ctx,beta[g][m][h],alpha[a][k][b-1]);

								e1 = Z3_mk_or(ctx,4,arg2);

								Z3_optimize_assert(ctx,opt,e1);
							}
						}
					}
				}
			}
		}
	}
	//cout<<"A";

	// 6)			
	for (a=0;a<F;a++)
	{
		for (b=1;b<fl[a].c;b++)
		{
			//fl[a].pathNode[b]; fl[a].pathPort[b];
			for (k=0;k<(H/fl[a].period);k++)
			{
				zi = 0;
				for (g=k*fl[a].period;g<(k*fl[a].period+fl[a].deadline);g++)
				{
					arg2[0] = Z3_mk_eq(ctx,alpha[a][k][b],mk_int(ctx,g));
					arg2[1] = BigA(ctx,g,k,a,b,fl[a].pathNode[b],fl[a].pathPort[b]);
					arg1[zi] = Z3_mk_and(ctx,2,arg2);
					zi ++;
				}
				e1 = Z3_mk_or(ctx,zi,arg1);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"B";

	// 7)
	for (a=0;a<F;a++)
	{
		for (k=0;k<(H/fl[a].period);k++)
		{
			for (b=1;b<fl[a].c;b++)
			{
				i = fl[a].pathNode[b];
				j = fl[a].pathPort[b];
				yi = 0;
				for (y=0;y<L;y++)
				{
					for(eachQ=0;eachQ<Q;eachQ++)
					{
						arg6[eachQ] = Z3_mk_eq(ctx,c[i][y][eachQ],q[a]);
					}
					arg4[1] = Z3_mk_or(ctx,Q,arg6);
					arg5[0] = Z3_mk_le(ctx,alpha[a][k][b],t[i][y]);
					arg5[1] = Z3_mk_le(ctx,t[i][y],beta[a][k][b]);
					arg4[0] = Z3_mk_and(ctx,2,arg5);
					arg3[2] = Z3_mk_and(ctx,2,arg4);
					arg3[0] = Z3_mk_lt(ctx,t[i][y],alpha[a][k][b]);
					arg3[1] = Z3_mk_lt(ctx,beta[a][k][b],t[i][y]);
					arg2[2] = Z3_mk_or(ctx,3,arg3); 
					arg2[0] = Z3_mk_eq(ctx,t[i][y],zero);
					arg2[1] = Z3_mk_eq(ctx,v[i][y][j],zero);		
					arg1[yi] = Z3_mk_or(ctx,3,arg2);
					yi ++;
				}

				e1 = Z3_mk_and(ctx,yi,arg1);
				//e1 = Z3_mk_eq(ctx,t[i][y],zero);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"C";

	//lambda[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"lambda_%d_%d",eachSW,eachL);
			lambda[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		yi = 0;
		for (y=0;y<L;y++)
		{
			arg1[0] = Z3_mk_eq(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],zero);
			arg2[0] = Z3_mk_and(ctx,2,arg1);

			arg1[0] = Z3_mk_gt(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],one);
			arg2[1] = Z3_mk_and(ctx,2,arg1);

			arg3[yi] = Z3_mk_or(ctx,2,arg2);
			yi ++;
		}
		e1 = Z3_mk_and(ctx,yi,arg3);
		Z3_optimize_assert(ctx,opt,e1);
	}

	sprintf_s (str,"ebar");
	ebar = mk_int_var(ctx,str);

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (y=0;y<L;y++)
		{
			arg1[y] = lambda[eachSW][y];
		}
		e1 = Z3_mk_add(ctx,y,arg1);
		Z3_optimize_assert(ctx,opt,Z3_mk_le(ctx,e1,ebar));
	}

	Z3_optimize_minimize(ctx,opt,ebar);
    //Z3_solver_assert(ctx, s, x_xor_y);

    printf("Z3 optimization is running...\n");
	//check(ctx, s, Z3_L_TRUE);
	r = Z3_optimize_check(ctx,opt,0,arg1);
	switch(r)
	{
	case Z3_L_TRUE:
		mr = Z3_optimize_get_model(ctx,opt);
		if (mr) Z3_model_inc_ref(ctx,mr);
		//display_model(ctx,stdout,mr);
		Z3_optimize_dec_ref(ctx, opt);
		Z3_del_context(ctx);
		return 1;
		break;
	default:
		printf ("err");
	}


    Z3_optimize_dec_ref(ctx, opt);
    Z3_del_context(ctx);

#ifdef _LOG_Z3_CALLS_
    Z3_close_log();
#endif

	return 0;
}

Z3_ast BigA(Z3_context ctx, int x,int k, int a,int b, int i, int j)
{
	int h,zi;
	Z3_ast arg1[MAXA],arg2[2],arg3[3];

	//return Z3_mk_eq(ctx,mk_int(ctx,1),mk_int(ctx,1)); //****

	zi = 0;
	for (h=k*fl[a].period;h<x;h++)
	{
		arg2[0] = Z3_mk_le(ctx,mk_int(ctx,h),beta[a][k][b-1]);
		arg2[1] = Z3_mk_not(ctx,BigB(ctx,h,i,j,a));
		arg1[zi] = Z3_mk_or(ctx,2,arg2);
		zi ++;
	}
	arg3[0] = Z3_mk_and(ctx,zi,arg1);
	arg3[1] = Z3_mk_gt(ctx,mk_int(ctx,x),beta[a][k][b-1]);
	arg3[2] = BigB(ctx,x,i,j,a);

	return Z3_mk_and(ctx,3,arg3);
}

Z3_ast BigB(Z3_context ctx, int x,int i,int j,int a)
{
	int y,yi,z,zi,eachQ;
	Z3_ast arg1[MAXA],arg2[2],arg3[3],arg4[MAXA],arg5[4],arg6[2];

	//return Z3_mk_eq(ctx,mk_int(ctx,1),mk_int(ctx,1)); //****
	yi = 0;
	for (y=0;y<L;y++)
	{
		//arg1[yi]
		arg2[0] = Z3_mk_eq(ctx,t[i][y],mk_int(ctx,0));
		arg3[0] = Z3_mk_eq(ctx,v[i][y][j],mk_int(ctx,1));
		
		for (eachQ=0;eachQ<Q;eachQ++)
			arg4[eachQ] = Z3_mk_eq(ctx,c[i][y][eachQ],q[a]);

		arg3[1] = Z3_mk_or(ctx,Q,arg4);
		
		zi = 0;
		for (z=0;z<L;z++)
		{
			if (z!=y)
			{
				arg5[0] = Z3_mk_eq(ctx,t[i][z],mk_int(ctx,0));
				arg5[1] = Z3_mk_eq(ctx,v[i][z][j],mk_int(ctx,0));
				arg6[0] = Z3_mk_lt(ctx,t[i][z],t[i][y]);
				arg6[1] = Z3_mk_le(ctx,t[i][y],mk_int(ctx,x));
				arg5[2] = Z3_mk_and(ctx,2,arg6);
				arg5[3] = Z3_mk_gt(ctx,t[i][z],mk_int(ctx,x));
				arg4[zi] = Z3_mk_or(ctx,4,arg5);
				//arg4[zi] = Z3_mk_or(ctx,3,arg5);
				zi ++;
			}
		}
		arg3[2] = Z3_mk_and(ctx,zi,arg4); 

		arg2[1] = Z3_mk_and(ctx,3,arg3);
		arg1[yi] = Z3_mk_or(ctx,2,arg2);
		yi++;
	}

	return Z3_mk_or(ctx,yi,arg1);
}

Z3_ast BigAOptDev(Z3_context ctx, int x,int k, int a,int b, int i, int j)
{
	int h,zi;
	Z3_ast arg1[MAXA],arg2[2],arg3[3];

	//return Z3_mk_eq(ctx,mk_int(ctx,1),mk_int(ctx,1)); //****

	zi = 0;
	for (h=k*fl[a].period;h<x;h++)
	{
		arg2[0] = Z3_mk_le(ctx,mk_int(ctx,h),beta[a][k][b-1]);
		arg2[1] = Z3_mk_not(ctx,BigBOptDev(ctx,h,i,j,a));
		arg1[zi] = Z3_mk_or(ctx,2,arg2);
		zi ++;
	}
	arg3[0] = Z3_mk_and(ctx,zi,arg1);
	arg3[1] = Z3_mk_gt(ctx,mk_int(ctx,x),beta[a][k][b-1]);
	arg3[2] = BigBOptDev(ctx,x,i,j,a);

	return Z3_mk_and(ctx,3,arg3);
}

Z3_ast BigBOptDev(Z3_context ctx, int x,int i,int j,int a)
{
	int y,yi,z,zi,eachQ;
	int boundL;
	Z3_ast arg1[MAXA],arg2[2],arg3[3],arg4[MAXA],arg5[4],arg6[2];

	//return Z3_mk_eq(ctx,mk_int(ctx,1),mk_int(ctx,1)); //****
	yi = 0;
	if (tCUp[i]+2*BJ>L)
		boundL = L;
	else
		boundL = tCUp[i]+2*BJ;
	for (y=0;y<boundL;y++)
	{
		//arg1[yi]
		arg2[0] = Z3_mk_eq(ctx,t[i][y],mk_int(ctx,0));
		arg3[0] = Z3_mk_eq(ctx,v[i][y][j],mk_int(ctx,1));
		
		for (eachQ=0;eachQ<Q;eachQ++)
			arg4[eachQ] = Z3_mk_eq(ctx,c[i][y][eachQ],q[a]);

		arg3[1] = Z3_mk_or(ctx,Q,arg4);
		
		zi = 0;
		//for (z=0;z<y;z++)
		for (z=0;z<boundL;z++)
		{
			if (z!=y)
			{
				arg5[0] = Z3_mk_eq(ctx,t[i][z],mk_int(ctx,0));
				arg5[1] = Z3_mk_eq(ctx,v[i][z][j],mk_int(ctx,0));
				arg6[0] = Z3_mk_lt(ctx,t[i][z],t[i][y]);
				arg6[1] = Z3_mk_le(ctx,t[i][y],mk_int(ctx,x));
				arg5[2] = Z3_mk_and(ctx,2,arg6);
				arg5[3] = Z3_mk_gt(ctx,t[i][z],mk_int(ctx,x));
				arg4[zi] = Z3_mk_or(ctx,4,arg5);
				//arg4[zi] = Z3_mk_or(ctx,3,arg5);
				zi ++;
			}
		}
		arg3[2] = Z3_mk_and(ctx,zi,arg4); 

		arg2[1] = Z3_mk_and(ctx,3,arg3);
		arg1[yi] = Z3_mk_or(ctx,2,arg2);
		yi++;
	}

	return Z3_mk_or(ctx,yi,arg1);
}

int Dev()
{
	int eachF,eachK,eachN;
	int eachsp1,eachsp2,temp;
	int mindeadline;
	//int spstart,spend;

	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
				alphaC[eachF][eachK][eachN] = betaC[eachF][eachK][eachN] = 0;
		}
	}

	AssignQFirstT(qC);

	for (eachN=0;eachN<SW;eachN++)
		tCUp[eachN] = 0;

	numofpacket = 0;
	sumoftableSMT = 0;
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			spf[numofpacket] = eachF;
			spk[numofpacket] = eachK;
			release[numofpacket] = fl[eachF].period * eachK;
			deadline[numofpacket] = release[numofpacket] + fl[eachF].deadline;
			numofpacket ++;
		}
	}

	for (eachsp1=0;eachsp1<numofpacket;eachsp1++)
	{
		sp[eachsp1] = eachsp1;
	}

	mindeadline = MAX;
	for (eachsp1=0;eachsp1<(numofpacket-1);eachsp1++)
	{
		for (eachsp2=eachsp1+1;eachsp2<numofpacket;eachsp2++)
		{
			if (deadline[sp[eachsp1]]>deadline[sp[eachsp2]])
			{
				temp = sp[eachsp1];
				sp[eachsp1] = sp[eachsp2];
				sp[eachsp2] = temp;
			}
		}
	}
	
	/*for (eachsp1=0;eachsp1<numofpacket;eachsp1++)
	{
		cout<<"flow "<<spf[sp[eachsp1]]<<", the "<<spk[sp[eachsp1]]<<"-th packet, "<<release[sp[eachsp1]]<<" "<<deadline[sp[eachsp1]]<<endl;
	}*/
	//ctx     = mk_context();
	Z31 = clock();
	for (eachsp1=0;eachsp1<numofpacket;) //[spstart,spend]
	{
		aC = eachsp1;
		bC = eachsp1+BJ-1;
		if (bC>(numofpacket-1))
			bC = numofpacket-1;

		if (!SMTTSNOptDev()) //qC, aC, bC
		{
			//Z3_del_context(ctx);
			Z32 = clock();
			return 0;
		}

		eachsp1 += BJ;
	}
	//Z3_del_context(ctx);
	Z32 = clock();
	Z3En = 1;
	for (eachN=0;eachN<SW;eachN++)
	{
		if (tCUp[eachN]>Z3En)
			Z3En = tCUp[eachN];
	}
	return 1;
}

int SMTTSNOptDev()
{
	char str[100];
	Z3_optimize opt;
	int eachF,eachN,eachSW,eachL,eachP,eachQ,eachK,eachA;
	int z,zi,b,a,g,h,m,k,i,j,y,yi;
	Z3_ast e1,e2;
	Z3_ast arg1[MAXA],arg2[MAXA],arg3[MAXA],arg4[MAXA],arg5[MAXA],arg6[Q];
	Z3_ast one,zero;
	Z3_model mr=0;
	Z3_lbool r;
	Z3_context ctx;
	//Z3_optimize opt;

#ifdef _LOG_Z3_CALLS_
    Z3_open_log("z3.log");
    LOG_MSG("SMT TSN Opt Dev");
#endif

	//printf("\nfind_model_example1\n");
	ctx     = mk_context();
	opt = Z3_mk_optimize(ctx);

	one = mk_int(ctx,1);
	zero = mk_int(ctx,0);

	//q[F]
	for (eachF=0;eachF<F;eachF++)
	{
		sprintf_s (str,"q_%d",eachF);
		q[eachF] = mk_int_var(ctx,str);
	}

	//assumptions[F+2*N*K]
	for (eachA=0;eachA<(F+2*N*K*F);eachA++)
	{
		sprintf_s (str,"assumption_%d",eachA);
		assumptions[eachA] = mk_int_var(ctx,str);
	}

	//t[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"t_%d_%d",eachSW,eachL);
			t[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	//v[SW][L][NUMPORT]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				sprintf_s (str,"v_%d_%d_%d",eachSW,eachL,eachP);
				v[eachSW][eachL][eachP] = mk_int_var(ctx,str);
			}
		}
	}

	//c[SW][L][Q]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				sprintf_s (str,"c_%d_%d_%d",eachSW,eachL,eachQ);
				c[eachSW][eachL][eachQ] = mk_int_var(ctx,str);
			}
		}
	}

	//alpha[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"alpha_%d_%d_%d",eachF,eachK,eachN);
				alpha[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	//beta[F][K][N]
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<K;eachK++)
		{
			for (eachN=0;eachN<N;eachN++)
			{
				sprintf_s (str,"beta_%d_%d_%d",eachF,eachK,eachN);
				beta[eachF][eachK][eachN] = mk_int_var(ctx,str);
			}
		}
	}

	// 1)
	//cout<<"1";
	for (eachSW=0;eachSW<SW;eachSW++)
		Z3_optimize_assert(ctx,opt,Z3_mk_eq(ctx,t[eachSW][0],one));
	
	//cout<<"2";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=1;eachL<L;eachL++)
		{
			zi = 0;
			for (z=(eachL+1);z<L;z++)
			{
				arg1[zi] = Z3_mk_eq(ctx,t[eachSW][z],zero);
				zi ++;
			}
			arg2[0] = Z3_mk_and(ctx,zi,arg1);
			arg2[1] = Z3_mk_eq(ctx,t[eachSW][eachL],zero);
			arg4[0] = Z3_mk_and (ctx,2,arg2);

			arg3[0] = Z3_mk_le(ctx,one,t[eachSW][eachL-1]);
			arg3[1] = Z3_mk_lt(ctx,t[eachSW][eachL-1],t[eachSW][eachL]);
			arg3[2] = Z3_mk_le(ctx,t[eachSW][eachL],mk_int(ctx,H));
			arg3[3] = Z3_mk_gt(ctx,t[eachSW][eachL],zero);
			arg4[1] = Z3_mk_and (ctx,4,arg3);

			e1 = Z3_mk_or (ctx,2,arg4);

			Z3_optimize_assert(ctx,opt,e1);
		}
	}
	
	//cout<<"3";

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				arg1[0] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],mk_int(ctx,(eachQ+1)));
				arg1[1] = Z3_mk_eq(ctx,c[eachSW][eachL][eachQ],zero);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	
	//cout<<"4";
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				arg1[0] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],zero);
				arg1[1] = Z3_mk_eq(ctx,v[eachSW][eachL][eachP],one);
				e1 = Z3_mk_or(ctx,2,arg1);
				Z3_optimize_assert(ctx,opt,e1);
			}
		}
	}
	//cout<<"5";

	/*for (eachF=0;eachF<F;eachF++)
	{
		arg1[0] = Z3_mk_le(ctx,one,q[eachF]);
		arg1[1] = Z3_mk_le(ctx,q[eachF],mk_int(ctx,Q));
		e1 = Z3_mk_and(ctx,2,arg1);
		Z3_optimize_assert(ctx,opt,e1);
	}
	cout<<"6";*/

	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			if (BeforebC(eachF,eachK)&& !(BeforeaC(eachF,eachK)))
			{
				for (b=0;b<fl[eachF].c;b++)
				{
					arg1[0] = Z3_mk_le(ctx,mk_int(ctx,eachK*fl[eachF].period),alpha[eachF][eachK][b]);
					arg1[1] = Z3_mk_le(ctx,alpha[eachF][eachK][b],beta[eachF][eachK][b]);
					arg1[2] = Z3_mk_le(ctx,beta[eachF][eachK][b],mk_int(ctx,eachK*fl[eachF].period+fl[eachF].deadline-1));
					e1 = Z3_mk_and(ctx,3,arg1);
					Z3_optimize_assert(ctx,opt,e1);
				
					arg2[0] = alpha[eachF][eachK][b];
					arg2[1] = mk_int(ctx,fl[eachF].size-1);
					e1 = Z3_mk_eq(ctx,beta[eachF][eachK][b],Z3_mk_add(ctx,2,arg2));
					Z3_optimize_assert(ctx,opt,e1);
				}
			}
		}
	}
	//cout<<"7";

	// 2)
	for (eachF=0;eachF<F;eachF++)
	{
		for (eachK=0;eachK<(H/fl[eachF].period);eachK++)
		{
			if (BeforebC(eachF,eachK)&&(!BeforeaC(eachF,eachK)))
			{
				for (b=0;b<(fl[eachF].c-1);b++)
				{
					e1 = Z3_mk_lt(ctx,beta[eachF][eachK][b],alpha[eachF][eachK][b+1]);
					Z3_optimize_assert(ctx,opt,e1);
				}
			}
		}
	}
	//cout<<"8";

	// 3) => 1)

	// 4) .....  change symbols to follow those in mu paper
	for (a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						if (BeforebC(a,k) && !BeforeaC(a,k) && BeforebC(g,m))
						{
							for (b=0;b<fl[a].c;b++)
							{
								for (h=0;h<fl[g].c;h++)
								{
									arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
									arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
									e1 = Z3_mk_and(ctx,2,arg1);
									arg2[0] = Z3_mk_not(ctx,e1);

									arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b+1]),mk_int(ctx,fl[g].pathNode[h+1]));
									arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b+1]),mk_int(ctx,fl[g].pathPort[h+1]));
									e2 = Z3_mk_and(ctx,2,arg1);
									arg2[1] = Z3_mk_not(ctx,e2);

									arg2[2] = Z3_mk_gt(ctx,alpha[a][k][b],beta[g][m][h]);
									arg2[3] = Z3_mk_gt(ctx,alpha[g][m][h],beta[a][k][b]);

									e1 = Z3_mk_or(ctx,4,arg2);

									Z3_optimize_assert(ctx,opt,e1);
								}
							}
						}
					}
				}
			}
		}
	}
	//cout<<"9";

	// 5) ......
	for(a=0;a<F;a++)
	{
		for (g=0;g<F;g++)
		{
			if (a!=g)
			{
				for (k=0;k<(H/fl[a].period);k++)
				{
					for (m=0;m<(H/fl[g].period);m++)
					{
						if (BeforebC(a,k) && !BeforeaC(a,k) && BeforebC(g,m))
						{
							for (b=1;b<fl[a].c;b++)
							{
								for (h=1;h<fl[g].c;h++)
								{
									arg1[0] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathNode[b]),mk_int(ctx,fl[g].pathNode[h]));
									arg1[1] = Z3_mk_eq(ctx,mk_int(ctx,fl[a].pathPort[b]),mk_int(ctx,fl[g].pathPort[h]));
									e1 = Z3_mk_and(ctx,2,arg1);
									arg2[0] = Z3_mk_not(ctx,e1);

									arg2[1] = Z3_mk_not(ctx,Z3_mk_eq(ctx,q[a],q[g]));

									arg2[2] = Z3_mk_lt(ctx,beta[a][k][b],alpha[g][m][h-1]);
									arg2[3] = Z3_mk_lt(ctx,beta[g][m][h],alpha[a][k][b-1]);

									e1 = Z3_mk_or(ctx,4,arg2);

									Z3_optimize_assert(ctx,opt,e1);
								}
							}
						}
					}
				}
			}
		}
	}
	//cout<<"A";

	// 6)			
	for (a=0;a<F;a++)
	{
		for (b=1;b<fl[a].c;b++)
		{
			//fl[a].pathNode[b]; fl[a].pathPort[b];
			for (k=0;k<(H/fl[a].period);k++)
			{
				if (BeforebC(a,k) && !BeforeaC(a,k))
				{
					zi = 0;
					for (g=k*fl[a].period;g<(k*fl[a].period+fl[a].deadline);g++)
					{
						arg2[0] = Z3_mk_eq(ctx,alpha[a][k][b],mk_int(ctx,g));
						arg2[1] = BigAOptDev(ctx,g,k,a,b,fl[a].pathNode[b],fl[a].pathPort[b]);
						arg1[zi] = Z3_mk_and(ctx,2,arg2);
						zi ++;
					}
					e1 = Z3_mk_or(ctx,zi,arg1);
					Z3_optimize_assert(ctx,opt,e1);
				}
			}
		}
	}
	//cout<<"B";

	// 7)
	for (a=0;a<F;a++)
	{
		for (k=0;k<(H/fl[a].period);k++)
		{
			if (BeforebC(a,k) && !BeforeaC(a,k))
			{
				for (b=1;b<fl[a].c;b++)
				{
					i = fl[a].pathNode[b];
					j = fl[a].pathPort[b];
					yi = 0;
					for (y=0;y<L;y++)
					{
						for(eachQ=0;eachQ<Q;eachQ++)
						{
							arg6[eachQ] = Z3_mk_eq(ctx,c[i][y][eachQ],q[a]);
						}
						arg4[1] = Z3_mk_or(ctx,Q,arg6);
						arg5[0] = Z3_mk_le(ctx,alpha[a][k][b],t[i][y]);
						arg5[1] = Z3_mk_le(ctx,t[i][y],beta[a][k][b]);
						arg4[0] = Z3_mk_and(ctx,2,arg5);
						arg3[2] = Z3_mk_and(ctx,2,arg4);
						arg3[0] = Z3_mk_lt(ctx,t[i][y],alpha[a][k][b]);
						arg3[1] = Z3_mk_lt(ctx,beta[a][k][b],t[i][y]);
						arg2[2] = Z3_mk_or(ctx,3,arg3); 
						arg2[0] = Z3_mk_eq(ctx,t[i][y],zero);
						arg2[1] = Z3_mk_eq(ctx,v[i][y][j],zero);		
						arg1[yi] = Z3_mk_or(ctx,3,arg2);
						yi ++;
					}

					e1 = Z3_mk_and(ctx,yi,arg1);
					//e1 = Z3_mk_eq(ctx,t[i][y],zero);
					Z3_optimize_assert(ctx,opt,e1);
				}
			}
		}
	}
	//cout<<"C";

	//lambda[SW][L]
	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (eachL=0;eachL<L;eachL++)
		{
			sprintf_s (str,"lambda_%d_%d",eachSW,eachL);
			lambda[eachSW][eachL] = mk_int_var(ctx,str);
		}
	}

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		yi = 0;
		for (y=0;y<L;y++)
		{
			arg1[0] = Z3_mk_eq(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],zero);
			arg2[0] = Z3_mk_and(ctx,2,arg1);

			arg1[0] = Z3_mk_gt(ctx,t[eachSW][y],zero);
			arg1[1] = Z3_mk_eq(ctx,lambda[eachSW][y],one);
			arg2[1] = Z3_mk_and(ctx,2,arg1);

			arg3[yi] = Z3_mk_or(ctx,2,arg2);
			yi ++;
		}
		e1 = Z3_mk_and(ctx,yi,arg3);
		Z3_optimize_assert(ctx,opt,e1);
	}

	sprintf_s (str,"ebar");
	ebar = mk_int_var(ctx,str);

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		for (y=0;y<L;y++)
		{
			arg1[y] = lambda[eachSW][y];
		}
		e1 = Z3_mk_add(ctx,y,arg1);
		Z3_optimize_assert(ctx,opt,Z3_mk_le(ctx,e1,ebar));
	}

	Z3_optimize_minimize(ctx,opt,ebar);
    //Z3_solver_assert(ctx, s, x_xor_y);

	for (eachF=0;eachF<F;eachF++)
		assumptions[eachF] = Z3_mk_eq(ctx,q[eachF],mk_int(ctx,qC[eachF]));
	numofas = F;

	for (eachP=0;eachP<aC;eachP++)
	{
		a = spf[sp[eachP]];
		g = spk[sp[eachP]];
		for (b=1;b<fl[a].c;b++)
		{
			assumptions[numofas] = Z3_mk_eq(ctx,alpha[a][g][b],mk_int(ctx,alphaC[a][g][b]));
			numofas ++;
			assumptions[numofas] = Z3_mk_eq(ctx,beta[a][g][b],mk_int(ctx,betaC[a][g][b]));
			numofas ++;
		}
	}

	for (eachSW=0;eachSW<SW;eachSW++)
	{
		//tCUp[eachSW]
		for (y=0;y<tCUp[eachSW];y++)
		{
			assumptions[numofas] = Z3_mk_eq(ctx,t[eachSW][y],mk_int(ctx,tC[eachSW][y]));
			numofas++;
			for (eachP=0;eachP<=NUMPORT;eachP++)
			{
				assumptions[numofas] = Z3_mk_eq(ctx,v[eachSW][y][eachP],mk_int(ctx,vC[eachSW][y][eachP]));
				numofas++;
			}
			for (eachQ=0;eachQ<Q;eachQ++)
			{
				assumptions[numofas] = Z3_mk_eq(ctx,c[eachSW][y][eachQ],mk_int(ctx,cC[eachSW][y][eachQ]));
				numofas++;
			}
		}
	}

    printf("Z3 optimization is running...\n");
	//check(ctx, s, Z3_L_TRUE);
	r = Z3_optimize_check(ctx,opt,numofas,assumptions);
	switch(r)
	{
	case Z3_L_TRUE:
		mr = Z3_optimize_get_model(ctx,opt);
		if (mr) Z3_model_inc_ref(ctx,mr);
		//display_model(ctx,stdout,mr);
		Z3_ast val;
		for (eachP=aC;eachP<=bC;eachP++)
		{
			a = spf[sp[eachP]];
			g = spk[sp[eachP]];
			//long long temp;
			for (b=1;b<fl[a].c;b++)
			{
				Z3_model_eval(ctx,mr,alpha[a][g][b],1,&val);
				Z3_get_numeral_int(ctx,val,&alphaC[a][g][b]);

				Z3_model_eval(ctx,mr,beta[a][g][b],1,&val);
				Z3_get_numeral_int(ctx,val,&betaC[a][g][b]);
			}
			//cout<<"K";
		}
		//cout<<"J";

		for (eachSW=0;eachSW<SW;eachSW++)
		{
			for (y=0;y<L;y++)
			{
				Z3_model_eval(ctx,mr,t[eachSW][y],1,&val);
				Z3_get_numeral_int(ctx,val,&tC[eachSW][y]);
				if (tC[eachSW][y]==0)
					break;
				else
				{
					for (eachP=0;eachP<=NUMPORT;eachP++)
					{
						Z3_model_eval(ctx,mr,v[eachSW][y][eachP],1,&val);
						Z3_get_numeral_int(ctx,val,&vC[eachSW][y][eachP]);
					}
					for (eachQ=0;eachQ<Q;eachQ++)
					{
						Z3_model_eval(ctx,mr,c[eachSW][y][eachQ],1,&val);
						Z3_get_numeral_int(ctx,val,&cC[eachSW][y][eachQ]);
					}
				}
			}
			tCUp[eachSW] = y;
		}
		Z3_model_eval(ctx,mr,ebar,1,&val);
		//cout<<"I";
		Z3_get_numeral_int(ctx,val,&sumoftableSMT);
		//cout<<"H";
		break;
	default:
		Z3_optimize_dec_ref(ctx, opt);
		Z3_del_context(ctx);
		return 0;
	}
	//cout<<"L";
    Z3_optimize_dec_ref(ctx, opt);
	//cout<<"M";
    Z3_del_context(ctx);

#ifdef _LOG_Z3_CALLS_
    Z3_close_log();
#endif

	return 1;
}

#endif

int NoWaitingAna()
{
	int eF,eK,eF2,eH,eH2;
	int x,prex;
	int r;

	r = 1;
	for (eF=0;eF<F;eF++)
	{
		for (eK=0;eK<K;eK++)
		{
			delayNWEDFAna[eF][eK] = 0;
			if (eK<(H/fl[eF].period))
			{
				releaseTime[eF][eK] = eK * fl[eF].period;
				deadlineTime[eF][eK] = releaseTime[eF][eK] + fl[eF].deadline;
			}
			else
				releaseTime[eF][eK] = deadlineTime[eF][eK] = MAX;
		}
	}

	for (eF=0;eF<F;eF++)
		for (eF2=0;eF2<F;eF2++)
			mu1[eF][eF2] = mu2[eF][eF2] = MAX;

	for (eF=0;eF<F;eF++)
	{
		for (eF2=0;eF2<F;eF2++)
		{
			if (eF!=eF2)
			{
				//mu1[eF][eF2]
				for (eH=0;eH<fl[eF].c;eH++)
				{
					for (eH2=0;eH2<fl[eF2].c;eH2++)
					{
						if ((fl[eF].pathNode[eH]==fl[eF2].pathNode[eH2])&& (fl[eF].pathNode[eH+1]==fl[eF2].pathNode[eH2+1]))
						{
							mu1[eF][eF2] = fl[eF].size * eH;
							break;
						}
					}
					if (eH2!=fl[eF2].c)
						break;
				}
				//mu2[eF][eF2]
				for (eH=(fl[eF].c-1);eH>=0;eH--)
				{
					for (eH2=(fl[eF2].c-1);eH2>=0;eH2--)
					{
						if ((fl[eF].pathNode[eH]==fl[eF2].pathNode[eH2])&& (fl[eF].pathNode[eH+1]==fl[eF2].pathNode[eH2+1]))
						{
							mu2[eF][eF2] = fl[eF].size * (eH+1);
							break;
						}
					}
					if (eH2>=0)
						break;
				}
			}
		}
	}

	for (eF=0;eF<F;eF++)
	{
		for (eK=0;eK<(H/fl[eF].period);eK++)
		{
			prex = MAX;
			x = fl[eF].size * fl[eF].c;
			while (prex!=x)
			{
				prex = x;
				x = D(eF,eK,prex) + fl[eF].size * fl[eF].c;
				if (x>fl[eF].deadline)
				{
					x = prex = MAX;
					r = 0;
					//return 0;
				}
			}
			delayNWEDFAna[eF][eK] = x;
		}
	}
	return r;
}

int D(int a,int k, int x)
{
	int HB[F][K]={0},HA[F][K]={0},LB[F][K]={0},LA[F][K]={0};
	int g,m;
	int b1,b2,b3,b4;
	int sum;
	int t1,t2,t3,t4;
	int con;
	int i1,i2;
	int max;
	int start,end;

	for (g=0;g<F;g++)
	{
		if (mu1[a][g]==MAX)
			continue;
		else
		{
			for (m=0;m<(H/fl[g].period);m++) //ÿ�������������ļ���
			{
				// releaseTime[g][m]+mu1[g][a], releaseTime[g][m]+fl[g].deadline-(size*c-mu2[g][a])
				// releaseTime[a][k]+mu1[a][g], releaseTime[a][k]+x-(size*c-mu2[a][g]) 
				b1 = releaseTime[g][m]+mu1[g][a];
				b2 = releaseTime[g][m]+fl[g].deadline-(fl[g].size*fl[g].c-mu2[g][a]);
				b3 = releaseTime[a][k]+mu1[a][g];
				b4 = releaseTime[a][k]+x-(fl[a].size*fl[a].c-mu2[a][g]);
				if (Cap(b1,b2,b3,b4))
				{
					//priority: H L, release time:A B, 
					if (deadlineTime[a][k]<=deadlineTime[g][m])
					{
						if (releaseTime[a][k]<releaseTime[g][m])
							LA[g][m] = 1;
						else
							LB[g][m] = 1;
					}
					else
					{
						if (releaseTime[a][k]<releaseTime[g][m])
							HA[g][m] = 1;
						else
							HB[g][m] = 1;
					}
				}
			}
		}
	}

	sum = 0;
	max = 0;
	//HB
	for (g=0;g<F;g++)
	{
		for (m=0;m<(H/fl[g].period);m++)
		{
			t3 = releaseTime[g][m];
			t4 = deadlineTime[g][m];
			t1 = releaseTime[a][k];
			t2 = deadlineTime[a][k]; //releaseTime[a][k] + x - fl[a].size * fl[a].c;

			/*if ((HA[g][m]==1)||(HB[g][m]==1))
			{
				max = 0;
				for (i1=t3;i1<=t4;i1++)
				{
					con = Conflict(g,m,i1,a,k,i2);
					if (max<con)
						max = con;
				}*/
			if (HB[g][m]==1)
			{
				max = 0;
				if (fl[g].size*fl[g].c>(t4-t1))
				{
					start = fl[g].size*fl[g].c - (t4-t1);
					end = fl[g].size*fl[g].c;
				}
				else
				{
					start = 0;
					end = fl[g].size * fl[g].c;
				}
				for (i1=start;i1<end;i1++)
				{
					max += Conflict(g,m,i1,a,k,0);
				}
			}
			else if (HA[g][m]==1)
			{
				max = 0;
				start = 0;
				end = fl[g].size * fl[g].c;
				for (i1=start;i1<end;i1++)
				{
					max += Conflict(g,m,i1,a,k,0);
				}
			}
			else if (LA[g][m]==1)
			{
				max = 0;
				start = 0;
				end = fl[g].size * fl[g].c;
				for (i1=start+1;i1<end;i1++)
				{
					max += Conflict(g,m,i1,a,k,0);
				}
			}
			else if (LB[g][m]==1)
			{
				max = 0;
				if (fl[g].size*fl[g].c>(t2-t3))
				{
					start = 0; //fl[g].size*fl[g].c - (t4-t1);
					end = t2 - t3;
				}
				else
				{
					start = 0;
					end = fl[g].size * fl[g].c;
				}
				for (i1=1;i1<end;i1++)
				{
					max += Conflict(g,m,i1,a,k,0);
				}
			}
		}
		sum += max;
	}
	return sum;
}