#include <iostream>
#include <vector>

using namespace std;

typedef double Number;

class DataSet {       
  public:              
    int pos;
    vector<Number> data;

    DataSet(int pos) {
      this->pos = pos;
      this->data.resize(pos);
    }

    // vector<Number> zeros(int pos) {
    //   vector<Number> v(pos);
    //   // auto v = vector<Number>(pos);
    //   for (int i = 0; i < pos; i++) {
    //     v[i] = i+1;
    //     cout <<i<< endl;
    //   }
      
    //   return v;
    // }
};

int main() {
  DataSet d(4);
  // DataSet *d = new DataSet(4);
  cout << d.pos << "\n";
  for (int i = 0; i < d.pos; i++)  {
    cout << d.data[i] << endl;
  }
  
  // delete d;
  return 0;
}