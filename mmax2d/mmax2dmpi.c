// Programa: mmaxmpi.c -- versao 0.1 - Versao MPI
// Programador: Edson
// Data: 18/09/2003
// O Dialogo: Este programa le uma matriz de inteiros de TAM x TAM e divide a 
// matriz em p submatrizes de tamanho TAM/p x TAM. Envia para cada tarefa uma 
// submatriz de tamanho TAMMAX/p x TAM. Cada tarefa i computa a subsequencia 
// maxima, o prefixo e o sufixo maximo de cada subsequencia da matriz de 
// prefixos.Cada processador envia para o processador raiz uma matriz com
// 5 valores (prefixo maximo, intervalo esquerdo, subsequencia maxima, 
// intervalo direito, sufixo maximo) de cada uma das submatrizes.
// O processador raiz recebe todos esses valores e computa a submatriz
// maxima.
// Declaracao das bibliotecas utilizadas
#include<mpi.h>
#include<stdio.h> // printf
#include<string.h>
#include<math.h>
//#include<stdarg.h>
#include<stdlib.h>
// Declaracao das constantes globais
const unsigned int MSGTAG = 11; // valor arbitrario para a TAG

// Declaracao da funcoes
void LeNum(int * const);
void LeDados(int *,int);
void CompParam(int, int, int * const, int * const, int * const, int * const);
void ImprimeMatriz(int *, int, int, int);
void SomaPre(int *, const int *, int, int);
void CompSV(int *, const int *, int, int, int, int);
void CompValSub(const int *, int * const, int * const, int * const, int * const, int * const, int * const, int * const, int * 
const, int * const, int);   
void ImprimeVetor(const int *, int, int, int, int);
void CompDataProc(int *, const int *, int, int, int, int, int, int, int, int, int, int);
void CompVetorMSG(int *, const int *, int * const);
void CompVetorDadosC(int *, const int *, int, int, int);
void CompSol(const int *, int *, int * const, int);
void CompMaxLoc(const int *, int, int * const);
void Resultados(int, int, double, double, double, int);

// inicio da funcao principal
int main(int argc, char *argv[])
{
// declaracao das variaveis locais
   int rank, size, tam, tamSM;
   int g, h, l;
   int numelem, numlinhas, numcolunas, nelMSG = 0, nelSol = 0;
   int indexP, indexME, indexMD, indexS;
   int prefix, SomaE, subseq, SomaD, suffix;
   int submatriz = 0;
   int Sol;
   int DataProc[5];
   int *SubVetor;
   int *VetorMSG;
   int *VetorS;
   int *VetorDadosTC;
   int *VetorDadosC;
   int *VetorDados; // vetor dados
   int *A, *SPA;
   int root=0;
   double startC = 0.0, startCT = 0.0, startM = 0.0;
   double finishC = 0.0, finishCT = 0.0, finishM = 0.0;
   double totalC = 0.0, totalCT = 0.0, totalM = 0.0;

// Passo 1. Inicilizacao
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size); // numero de tarefas
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // identificacao da tarefa
// Passo 2. Leitura dos dados dos arquivos pela tarefa 0
   if (rank == root) {
// Passo 2.3. Dimensione o vetor de entrada
      LeNum(&numelem);
      //printf("%i\n",numelem);
      VetorDados = (int *)malloc(numelem*sizeof(int));
      LeDados(VetorDados,numelem);
   }
   MPI_Bcast(&numelem, 1, MPI_INT, root, MPI_COMM_WORLD);
   //printf("rank: %i tam: %i\n", rank, numelem);

// Passo 3. Envio os dados para as tarefas filhos
   CompParam(numelem, size, &tam, &numlinhas, &numcolunas, &tamSM);
// Passo 3.1. Dimensione o SubVetor
   A = (int *)malloc(tam*sizeof(int));
// Passo 3.2. Envie os dados as tarefas filhos
   MPI_Scatter(VetorDados, tam, MPI_INT, A, tam, MPI_INT, root, MPI_COMM_WORLD);
//   ImprimeMatriz(A, rank, numlinhas, numcolunas);
// Passo 3.3. Libere o espaço alocado a VetorDados
   if (rank == root) {
      free(VetorDados);
   }
// Inicio da tomada de tempo
   MPI_Barrier(MPI_COMM_WORLD);
   startC = MPI_Wtime();

// Passo 4. Compute a soma de prefixos na submatriz.
   SPA = (int *)malloc((numlinhas*(numcolunas+1))*sizeof(int));
   //printf("rank %i tam SPA %i\n", rank, numlinhas*(numcolunas+1));
   SomaPre(SPA, A, numlinhas, numcolunas);
   //ImprimeMatriz(SPA, rank, numlinhas, numcolunas+1);
  
// Passo 5. Compute os valores para subsequencia maxima de SubVetor(g,h)      
   SubVetor = (int *)malloc(numlinhas*sizeof(int));
   VetorMSG = (int *)malloc(5*tamSM*sizeof(int));

   for (h = 1; h < numcolunas+1; h++) {
      for (g = 1; g <= h; g++) {
	 CompSV(SubVetor, SPA, numlinhas, numcolunas, g, h);
// Passo P.2. Imprima o Vetor Recebido.
//         ImprimeVetor(SubVetor, numlinhas, rank, g, h); 
	 CompValSub(SubVetor, &prefix, &SomaE, &subseq, &SomaD, &suffix, &indexP, &indexME, &indexMD, &indexS, numlinhas);   
// Passo 5.6. Armazene os valores a serem enviados
         CompDataProc(DataProc, SubVetor, numlinhas, indexP, indexME, indexMD, indexS, prefix, SomaE, subseq, SomaD, suffix);
// Passo P.3. Imprima os valores a serem enviados
//         ImprimeVetor(DataProc, 5, rank, g, h); 
// Passo 5.7. Armazene os valores no vetor de MSG
         CompVetorMSG(VetorMSG, DataProc, &nelMSG);
//         printf("rank %i nelMSG=%i\n", rank, nelMSG);
      } // for g
   } //for h
// Passo P.3. Imprima os valores a serem enviados
   ImprimeMatriz(VetorMSG, rank, numlinhas, 5*(tamSM/numlinhas)); 
// Passo . Libere os vetores que não estão sendo utilizados
   
   free(SPA);
   free(SubVetor);

// Passo . Dimensione o vetor que ira armazenar as MSG's
   VetorDadosTC = (int *)malloc(5*tamSM*sizeof(int));
// Inicio da tomada de tempo da comunicacao
   MPI_Barrier(MPI_COMM_WORLD);
   startM = MPI_Wtime();
   MPI_Alltoall(VetorMSG, (5*tamSM)/size, MPI_INT, VetorDadosTC, (5*tamSM)/size, MPI_INT, MPI_COMM_WORLD);
// Passo . Libere


   free(VetorMSG);

// Passo . Imprima a Matriz VetorDadosTC
//   ImprimeMatriz(VetorDadosTC, rank, numlinhas, 5*(tamSM/numlinhas)); 

// Final da tomada de tempo da comunicacao
   MPI_Barrier(MPI_COMM_WORLD);
   finishM = MPI_Wtime();
   totalM = (finishM - startM);

// Passo 4. Receba as Somas Parciais

// Inicio da tomada de tempo (verificar)
//   MPI_Barrier(MPI_COMM_WORLD);
   startCT = finishM;

   VetorDadosC = (int *)malloc(size*5*sizeof(int)); 
   VetorS = (int *)malloc(tamSM*sizeof(int)); 

   for (l = 0; l < tamSM/size; l++) {
// Passo . Compute o vetor VetorDadosC
      CompVetorDadosC(VetorDadosC, VetorDadosTC, l, size, tamSM);
// Passo . Imprima o VetorDadosC
//     ImprimeVetor(VetorDadosC, 5*size, rank, l, l);
// Passo . Compute o VetorS
      CompSol(VetorDadosC, VetorS, &nelSol, size); 
   } // end l
// Passo . Compute a solucao local
   CompMaxLoc(VetorS, tamSM/size, &submatriz);
//   ImprimeVetor(VetorS, tamSM/size, rank, 1,1);
   //printf("rank %i max = %i\n", rank, submatriz);
// Passo . Compute a solucao Global
   MPI_Reduce(&submatriz, &Sol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
// Passo . Libere os vetores 

/*printf("pointers\n");
printf("%p\n",VetorDadosC);
printf("%p\n",VetorDadosTC);
printf("%p\n",VetorS);*/

//printf("her1\n");
   free(VetorDadosC);
   free(VetorDadosTC);
   free(VetorS);
//printf("her2\n");

// Final da tomada de tempo
   MPI_Barrier(MPI_COMM_WORLD); 
   finishCT = MPI_Wtime(); 
   totalCT = (finishCT - startCT); 

// Final da tomada de tempo
//   MPI_Barrier(MPI_COMM_WORLD); 
   finishC = finishCT; 
   totalC = (finishC - startC);
   
// Passo . Escreva os resultados
   if (rank == root) {
     //Resultados(numelem, size, totalC, totalM, totalCT, Sol); 
     printf("%.9lf\n",(finishCT - startC)*1000);
     printf("%d\n",Sol);
  }

// Passo 9. Finalize o MPI
   MPI_Finalize();
   return 0;
} // fim funcao main


// implementacao das funcoes

void LeNum(int * const numelem) {
// Declaracao das variaveis
   //FILE *ArqA;
   //int num;

// Passo 2.1. Abra os arquivos de entrada
   //ArqA = fopen("sequencia.txt", "r");
// Passo 2.2. Conte o numero de caracteres do ArquivoA.txt
   //fscanf(ArqA,"%i", &num);
   //*numelem = num;
   //fclose(ArqA);
   scanf("%d",numelem);
   *numelem *= *numelem;
}   

void LeDados(int *VetorDados, int num) {
// Declaracao das variaveis
   //FILE *ArqA;
   int i;
   //int num;

// Passo 2.1. Abra os arquivos de entrada
   //ArqA = fopen("sequencia.txt", "r");
// Passo 2.2. Conte o numero de caracteres do ArquivoA.txt
   //fscanf(ArqA,"%i", &num);
   for (i = 0; i < num; i++) {
      scanf("%d", &VetorDados[i]);
   }
   //fclose(ArqA);
// Passo P.1. Imprima o Vetor Recebido.
//   for (i = 0; i < num; i++) {
//      printf("%i ", VetorDados[i]);
//   }
//   printf("\n");
}   

void CompParam(int numelem, int size, int * const tam, int * const numlinhas, int * const numcolunas, int * const tamSM) {

// Passo 3.1. Dimensione o SubVetor
   *tam = (int) numelem/size; // tamanho subvetor
   *numcolunas = (int) sqrt(numelem);
   *numlinhas = (int) *numcolunas/size;
   *tamSM = (*numcolunas+1)*(*numcolunas/2);
}

void ImprimeMatriz(int *A, int rank, int numlinhas, int numcolunas) {
// Declaracao das variaveis locais
   //int i;
   //int j;

// Passo P.1. Imprima o Vetor Recebido.
   /*for (i = 0; i < numlinhas; i++) {
      printf("rank %i, ", rank);
      for (j = 0; j < numcolunas; j++) {
         printf("M[%i] = %i ", i*numcolunas+j, A[i*numcolunas + j]);
      }
      printf("\n");
   }*/
}

void SomaPre(int *SPA, const int *A, int numlinhas, int numcolunas) {
// Declaracao das variaveis locais
   int i;
   int j;

   for (i = 0; i < numlinhas; i++) {
      SPA[i*(numcolunas+1)] = 0;
      for (j = 1; j < numcolunas+1; j++) {
	 SPA[i*(numcolunas+1)+j] = SPA[i*(numcolunas+1)+j-1] + A[i*numcolunas+j-1];
      }
   }
}

void CompSV(int *SubVetor, const int *SPA, int numlinhas, int numcolunas, int g, int h) {
// Declaracao das variaveis locais
   int i;

   for (i = 1; i < numlinhas+1; i++) {
      SubVetor[i-1] = SPA[(i-1)*(numcolunas+1)+h] - SPA[(i-1)*(numcolunas+1)+g-1];
   }
}

void CompValSub(const int *SubVetor, int * const prefix, int * const SomaE,int * const subseq, int * const SomaD, int * const 
suffix, int * const indexP, int * const indexME, int * const indexMD, int * const indexS, int numlinhas) {   
// Declaracao das variaveis locais
   int i, j;
   int suffixA = 0;
   int SomaP = 0, SomaPD = 0;
   int *SubMax, *SufMax;

// Passo 5.2. Dimensione os vetores da Subsequencia maxima e Sufixo maximo
   SubMax = (int *)malloc(numlinhas*sizeof(int));
   SufMax = (int *)malloc(numlinhas*sizeof(int));
// Inicialize as variáveis
   *indexP = *indexME = *indexMD = *indexS = 0;
   *prefix = *SomaE = *subseq = *SomaD = *suffix = 0;
// Passo 5.3. Compute os valores a serem enviados
   j = 0;
   for (i = 0; i < numlinhas; i++) {
// Passo 5.4. Compute o sufixo maximo, subsequencia maxima e soma direita
      suffixA = *suffix;
      if (*suffix + SubVetor[i] > *subseq) {
         *suffix = *suffix + SubVetor[i];
         SufMax[j] = SubVetor[i];
         *indexME = *indexS;
         *indexMD = *indexME + j;
         *subseq = *suffix;
         j++;
         SomaPD = 0;
         *SomaD = 0;
      } 
      else if (*suffix + SubVetor[i] > 0) {
         SufMax[j] = SubVetor[i];
         *suffix = *suffix + SubVetor[i];
         j++;
	 SomaPD = SomaPD + SubVetor[i];
      } else {
         j = 0;
         *indexS = i+1;
         *suffix = 0;
         *SomaD = *SomaD + SomaPD + SubVetor[i];
         SomaPD = 0;
      }
// Passo 5.5. Compute o Prefixo maximo e Soma esquerda
      SomaP = SomaP + SubVetor[i];
      if (SomaP > *prefix) {
         *prefix = SomaP;
         *indexP = i;
         *SomaE = 0; 
      } else {
         *SomaE = *SomaE + SubVetor[i];
      }
   }
   *SomaE = *SomaE - *subseq - *SomaD - *suffix;
   free(SubMax);
   free(SufMax);

// Passo P.2. Imprima os valores computados
//   printf("(%i,%i,%i,%i,%i)\n", *prefix, *SomaE, *subseq, *SomaD, *suffix);
//   printf("indices = %i %i %i %i\n", *indexP, *indexME, *indexMD, *indexS);
}

void ImprimeVetor(const int *SubVetor, int numlinhas, int rank, int g, int h) {
// Declaracao das variaveis locais
   /*int i;

   printf("rank (%i,%i,%i) %i ", rank, h, g, numlinhas);
   for (i = 0; i < numlinhas; i++) {
      printf("%i ", SubVetor[i]);
   }
   printf("\n");*/
}

void CompDataProc(int *DataProc, const int *SubVetor, int numlinhas, int indexP, int indexME, int indexMD, int indexS, int 
prefix, int SomaE, int subseq, int SomaD, int suffix) {

   if (indexP < indexME) {
      DataProc[0] = prefix;
      DataProc[1] = SomaE;
   } else if (indexME > 0) {
      DataProc[1] = SubVetor[indexME-1];
      DataProc[0] = prefix - (subseq + DataProc[1]);
   } else {
      DataProc[0] = 0;
      DataProc[1] = 0;
   }
   DataProc[2] = subseq;
   if (indexS > indexMD) {
      DataProc[3] = SomaD;
      DataProc[4] = suffix;
   } else if (indexMD < numlinhas-1) {
      DataProc[3] = SubVetor[indexMD+1];
      DataProc[4] = suffix - (subseq + DataProc[3]);
   } else {
      DataProc[3] = 0;
      DataProc[4] = 0;
   }
}


void CompVetorMSG(int *VetorMSG, const int *DataProc, int *nelMSG) {
// Declaracao das variaveis locais
   int i;

   for (i = 0; i < 5; i++) {
      VetorMSG[*nelMSG] = DataProc[i];
      *nelMSG = *nelMSG+1;
   }
}

void CompVetorDadosC(int *VetorDadosC, const int *VetorDadosTC, int l, int size, int tamSM) {
// Declaracao da Variaveis locais
   int i, j;
      
   for (i = 0; i < size; i++) {
      for (j = 0; j < 5; j++) {
         VetorDadosC[(5*i)+j] = VetorDadosTC[(l*5)+i*((5*tamSM)/size)+j];
      }
   }
}

void CompSol(const int *VetorDadosC, int *VetorS, int * const nelSol, int size) {
// Declaracao das Variaveis locais
   int i, j = 0; 
   int suffix = 0, subseq = 0, indexS = 0;
   int *SubMaxT, *SufMaxT;
      
   SubMaxT = (int *)malloc(size*5*sizeof(int));
   SufMaxT = (int *)malloc(size*5*sizeof(int));
   for (i = 0; i < 5*size; i++) {
      if (suffix + VetorDadosC[i] > subseq) {
         suffix = suffix + VetorDadosC[i];                
         SufMaxT[j] = VetorDadosC[i];
         j++;
 indexS = j; // verificar !!!!!
         memcpy(SubMaxT, SufMaxT, (i - indexS + 1)*sizeof(int));
         subseq = suffix;
      } 
      else if (suffix + VetorDadosC[i] > 0) {
	 suffix = suffix + VetorDadosC[i];
	 SufMaxT[j] = VetorDadosC[i];
         j++;
      } else {
	 j = 0;
         indexS = i+1;
         suffix = 0;
      }
   } // end i
   VetorS[*nelSol] = subseq;
//   printf("seqmax %i\n", subseq);
   *nelSol = *nelSol+1;
   free(SubMaxT);
   free(SufMaxT);
}

void CompMaxLoc(const int *VetorS, int tamanho, int * const submatriz) {
// Declaracao das Variaveis locais
   int i;

   *submatriz = VetorS[0];
   for (i = 1; i < tamanho; i++) {
      if (VetorS[i] > *submatriz)
	 *submatriz = VetorS[i];
   }
}

void Resultados(int numelem, int size, double totalC, double totalM, double totalCT, int Sol) {
// Declaracao das Variaveis locais
   //FILE *ArqS;

// Passo 8.1. Abre o arquivo 
   //ArqS = fopen("ArquivoS.txt", "a"); 
// Passo 8.2. Escreve no arquivo
   //fprintf(ArqS, "%i %i %lf %lf %lf %i\n", numelem, size, totalC, totalM, totalCT, Sol);
      //fclose(ArqS);
   printf("%lf\n",totalC*1000);
} 

