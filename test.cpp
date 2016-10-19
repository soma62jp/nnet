/* 
 * nnet test
 * 2016.06.06
 * author:soma62jp
 *                  */


#include "test.h"

//using namespace std;


nnet::nnet(const int &inum,const int &hnum,const int &onum,const int &pnum):
	  inputnum(inum)
	 ,hiddennum(hnum)
	 ,outputnum(onum)
	 ,patternnum(pnum)
	 ,ALPHA(0.8)
{

	// 初期化
	Hi.resize(hiddennum + 1);           // bias項
	fill(Hi.begin(), Hi.end(), 0);

	Oi.resize(outputnum);
	fill(Oi.begin(), Oi.end(), 0);


	E_ih.resize(hiddennum);
	fill(E_ih.begin(), E_ih.end(), 0);

	E_ho.resize(outputnum);
	fill(E_ho.begin(), E_ho.end(), 0);


	// pattern*input
	Ii.resize(patternnum);				// ()内の数字が要素数になる
	for( int i=0; i<patternnum; i++ ){
		Ii[i].resize(inputnum);
		std::fill(Ii[i].begin(), Ii[i].end(), 0);
	}

	// pattern*output
	Ti.resize(patternnum);
	for(int i=0;i<patternnum;i++){
		Ti[i].resize(outputnum);
		std::fill(Ti[i].begin(), Ti[i].end(), 0);
	}

	// hidden*input
	W_ih.resize(hiddennum);
	for( int i=0; i<hiddennum; i++ ){
		W_ih[i].resize(inputnum + 1);     // bias項
		std::fill(W_ih[i].begin(), W_ih[i].end(), 0);
	}


	// output*hidden
	W_ho.resize(outputnum);
	for( int i=0; i<outputnum; i++ ){
		W_ho[i].resize(hiddennum + 1);    // bias項
		std::fill(W_ho[i].begin(), W_ho[i].end(), 0);
	}

	//initialize parameter
	// 重みのランダマイズ
	srand((unsigned int)time(NULL));

	for(int i=0;i<(int)W_ih.size();i++){
		for(int j=0;j<(int)W_ih[i].size();j++){
			W_ih[i][j]=urand();
		}
	}

	for(int i=0;i<(int)W_ho.size();i++){
		for(int j=0;j<(int)W_ho[i].size();j++){
			W_ho[i][j]=urand();
		}
	}

}

nnet::~nnet()
{

}

void  nnet::setAlpha(const double &value)
{
	ALPHA = value;
}

void nnet::setErrorEv(const double &value)
{
	ERROREV = value;
}

void  nnet::setInData(const int &pnum,const int &i,const double &value)
{
	if(pnum>=patternnum || i>=inputnum){
		//cout << "can't set Indata." << endl;
		outlog("can't set Indata.");
		return;
	}
	Ii[pnum][i] = value;
}

void nnet::setTeachData(const int &pnum,const int &i,const double &value)
{
	if(pnum>=patternnum || i>=outputnum){
		//cout << "can't set Teachdata." << endl;
		outlog("can't set Teachdata.");
		return;
	}
	Ti[pnum][i] = value;
}

void nnet::setPredictData(const int &i,const double &value)
{
	if(i>=inputnum){
		//cout << "can't set Predictdata." << endl;
		outlog("can't set Predictdata.");
		return;
	}
	Ii[patternnum-1][i] = value;
}


void nnet::foward_propagation(const int &pnum)
{
	int i,j;
	double sum;

	// 入力層ー＞隠れ層
	for(i=0;i<hiddennum;i++){
		sum=0;
		for(j=0;j<inputnum;j++){
			// 重み×入力値
			sum+=W_ih[i][j]*Ii[pnum][j];
		}
		sum-=W_ih[i][inputnum];						// bias項
		// 重み×入力値の総和にバイアス項を足してアクティベーション関数に通したものが中間層入力
		Hi[i] = activationFunc(sum);
	}

	// 隠れ層ー＞出力層
	for(i=0;i<outputnum;i++){
		sum=0;
		for(j=0;j<hiddennum;j++){
			// 重み×中間層入力
			sum+=W_ho[i][j]*Hi[j];
		}
		sum-=W_ho[i][hiddennum];						// bias項
		// 重み×中間層入力値の総和にバイアス項を足してアクティベーション関数に通したものが出力層
		Oi[i]=outputFunc(sum);
	}

}

void nnet::back_propagation(const int &pnum)
{

	int i,j,k;

	// 重みの変化量[隠れ層ー＞出力層]を計算
	// 隠れ層の学習信号を計算
	for(i=0;i<outputnum;i++){
		// 出力層での誤差信号=(教師信号-出力）* f'(出力)
		// f'(出力)はシグモイド関数の微分
		E_ho[i]=(Ti[pnum][i]-Oi[i]) * outputFunc_diff(Oi[i]);
		for(j=0;j<hiddennum;j++){
			// 隠れ層->出力層重み更新
			W_ho[i][j]+=ALPHA*E_ho[i]*Hi[j];
		}
		W_ho[i][hiddennum]+=ALPHA*(-1.0)*E_ho[i];		// bias項
	}


	// 重みの変化量[入力層ー＞隠れ層]を計算
	for(i=0;i<outputnum;i++){
		for(j=0;j<hiddennum;j++){
			// 中間層での誤差信号=f'(隠れ層出力) * 隠れ層->出力層重み * 隠れ層<-出力層での誤差
			E_ih[j] = activationFunc_diff(Hi[j])*W_ho[i][j]*E_ho[i];
			for(k=0;k<inputnum;k++){
				// 入力<-隠れ層での重み更新
				W_ih[j][k]+=ALPHA*Ii[pnum][k]*E_ih[j];
			}
			W_ih[j][inputnum]+=ALPHA*(-1.0)*E_ih[j];		// bias項
		}
	}


}

void nnet::train()
{
	double verror;
	unsigned long gen;
	int i,j,ip;
	const double delta_err = 10e-7;

#if 1
	// 状態表示
	std::string str;
	for(i=0;i<patternnum;i++){
		str="";
		for(j=0;j<inputnum;j++){
			str += tostr(Ii[i][j]);
			str += ",";
		}
		str += ":";
		for(j=0;j<outputnum;j++){
			str+=tostr(Ti[i][j]);
		}
		outlog(str);
	}
#endif

	// 学習開始
	gen = 0;
	verror = 20.0;
	while((verror > ERROREV) && (gen < MAXGEN))
	{
		gen++;

		// foward and back propagation
		for(i=0;i<patternnum;i++){
#if 0
			ip = patternnum * random();
#else
			ip=i;
#endif
			foward_propagation(ip);
			back_propagation(ip);
		}

		// error calc
		verror = 0;
		for(i=0;i<patternnum;i++){
			foward_propagation(i);
			for(j=0;j<outputnum;j++){
				// --誤差関数
				// https://github.com/tiny-dnn/tiny-dnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
				//verror+=pow((Ti[i][j] - Oi[j]) ,2.0) * 0.5;   // 二乗誤差
				//verror+= -Ti[i][j] * std::log(Oi[j]) - (1.0 - Ti[i][j]) * std::log(1.0 - Oi[j]);	// 交差エントロピー
				verror+= -Ti[i][j] * std::log(Oi[j] + delta_err); 												// 交差エントロピー（マルチクラス）
				// --
			}
		}
		verror /= (double)patternnum;

		//cout << verror << endl;
		outlog(verror);

	}
}

//void nnet::predict()
void nnet::predict(const int &pnum)
{
	int i;
	foward_propagation(patternnum-1);

#if 0
	std::string str;
	str="";
	for(i=0;i<outputnum;i++){
		str += tostr(Oi[i]);
		str += ",";
	}
	outlog(str);
#elif 0
	for(i=0;i<outputnum;i++){
		//cout << Oi[i] << endl;
		outlog(Oi[i]);
	}
#else
	// 状態表示
	std::string str;
	str="";
	for(i=0;i<outputnum;i++){
		str += tostr(Oi[i]);
		str += ",";
	}
	str += ":";
	for(i=0;i<outputnum;i++){
		str+=tostr(Ti[pnum][i]);
	}
	outlog(str);
#endif


}

double nnet::random()
{
	double r;
	int i;

	i = rand();
	if (i != 0){
		i--;
	}
	r = (double)i / RAND_MAX;
	return (r);
}

#if 1
//--sigmoid function
// alpha=0.8位
double nnet::activationFunc(double x)
{
	return (1/(1+exp(-(x))));
}

double nnet::activationFunc_diff(double x)
{
	return (x*(1-x));
}
//--
#elif 1
//--ReLU function(NG)
double nnet::activationFunc(double x)
{
	return ((x >= 0)?  x : 0);
}

double nnet::activationFunc_diff(double x)
{
	return ((x >= 0)?  1 : 0);
}
//--
#else
//--Tanh function
// alpha=0.01位
double nnet::activationFunc(double x)
{
	return tanh(x);
}

double nnet::activationFunc_diff(double x)
{
	return 1 - (tanh(x)*tanh(x));
}
//--
#endif

// 出力関数
//（例）ソフトマックス関数など
#if 1
double nnet::outputFunc(double x)
{
	return (1/(1+exp(-(x))));
}

double nnet::outputFunc_diff(double x)
{
	return (x*(1-x));
}
#else
double nnet::outputFunc(double x)
{
	return tanh(x);
}

double nnet::outputFunc_diff(double x)
{
	return 1 - (tanh(x)*tanh(x));
}
#endif

double nnet::urand()
{
	return ((double) rand()/RAND_MAX * (RHIGH - RLOW) + RLOW);
}

void nnet::outlog(double value)
{
	//Form1->Memo1->Lines->Add(FloatToStr(value));
	std::cout << value << std::endl;
}

void nnet::outlog(std::string str)
{
	//Form1->Memo1->Lines->Add(str.c_str());
	std::cout << str << std::endl;
}



int main()
{
  int i;
  double input[150][4];

  nnet net(4,4,3,150);

  net.setAlpha(0.08);
  net.setErrorEv(0.1);

  //ファイルの読み込み
	std::stringstream ss;
	std::ifstream ifs("iris.txt");
	if(!ifs){
        net.outlog("--  入力エラー --");
        return 0;
    }

    //csvファイルを1行ずつ読み込む
	std::string str;
	int inputnum;
	int cnt = 0;
	while(getline(ifs,str)){
		std::string token;
		std::istringstream stream(str);

		//1行のうち、文字列とコンマを分割する
		inputnum = 0;
		while(getline(stream,token,',')){
			//すべて文字列として読み込まれるため
			//数値は変換が必要
			if(inputnum < 4){
				//double temp=stof(token); //stof(string str) : stringをfloatに変換

				// 文字列から数値に変換
				double temp;
				ss << token;
				ss >> temp;

				input[cnt][inputnum] = temp;

				ss.clear(); // 状態をクリア.
				ss.str(""); // 文字列をクリア.

				net.setInData(cnt,inputnum,temp);
			}else{
				if(token=="s"){
					net.setTeachData(cnt,0,1);
					net.setTeachData(cnt,1,0);
					net.setTeachData(cnt,2,0);
				}else if(token=="e"){
					net.setTeachData(cnt,0,0);
					net.setTeachData(cnt,1,1);
					net.setTeachData(cnt,2,0);
				}else{
					net.setTeachData(cnt,0,0);
					net.setTeachData(cnt,1,0);
					net.setTeachData(cnt,2,1);
				}
			}
			inputnum++;
		}
		cnt++;
	}

  net.train();


  net.outlog("--  Predict --");

  for(i=0;i<150;i++){
	net.setPredictData(0,input[i][0]);
	net.setPredictData(1,input[i][1]);
	net.setPredictData(2,input[i][2]);
	net.setPredictData(3,input[i][3]);
	net.predict(i);
  }

  return 0;

}
