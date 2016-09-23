//---------------------------------------------------------------------------

#ifndef nnetH
#define nnetH

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <ctime>        // time
#include <cstdlib>      // srand,rand
#include <fstream>
#include <string>
#include <sstream> //文字ストリーム

//---------------------------------------------------------------------------
//using namespace std;

class nnet
{

	public:
		nnet(const int &inum,const int &hnum,const int &onum,const int &pnum);
		~nnet();
		void setAlpha(const double &value);
		void setErrorEv(const double &value);
		void setInData(const int &pnum,const int &i,const double &value);
		void setTeachData(const int &pnum,const int &i,const double &value);
		void setPredictData(const int &i,const double &value);
		void train();
		void predict(const int &pnum);
		//void predict();

		void outlog(std::string str);
		void outlog(double value);

	private:
		int inputnum;
		int hiddennum;
		int outputnum;
		int patternnum;
		std::vector< std::vector<double> > Ii;					// 入力層
		std::vector<double> Hi;								// 隠れ層
		std::vector<double> Oi;      						// 出力層
		std::vector< std::vector<double> > W_ih;					// 入力層->隠れ層重み(隠れ×入力)
		std::vector< std::vector<double> > W_ho;					// 隠れ層->出力層重み(出力×隠れ)

		// foward propagation
		std::vector< std::vector<double> > Ti;					// 教師信号

		// back propagation
		std::vector<double> E_ih;           					// 入力<-隠れ層での誤差
		std::vector<double> E_ho;           					// 隠れ層<-出力層での誤差

		// const parameter
		double ALPHA;
		double ERROREV;
		#define RLOW        -0.30
		#define RHIGH       0.30
		#define MAXGEN      200000

		double activationFunc(double x);
		double activationFunc_diff(double x);
		double outputFunc(double x);
		double outputFunc_diff(double x);

		double urand();

		// functions
		void foward_propagation(const int &pnum);
		void back_propagation(const int &pnum);
		double random() ;

		template <typename T> std::string tostr(const T& t)
		{
			std::ostringstream os; os<<t; return os.str();
		}



};

#endif
