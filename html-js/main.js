class DataSet {
    constructor({data, nro_in, 
                target, nro_out}) {
        this.data = data;
        this.nro_in = nro_in;
        this.nro_cases = data.length/nro_in;

        this.target = target;
        this.nro_out = nro_out;

        // auxiliar
        this.case = 0;
    }

    get x() {
        return this.data.slice(this.case*this.nro_in, this.case*this.nro_in+this.nro_in);
    }

    get t() {
        return this.target.slice(this.case*this.nro_out, this.case*this.nro_out+this.nro_out);
    }
}

class MLP {
    constructor({data, epochs=1000, alpha = 0.001}) {
        this.alpha = alpha;
        this.data = data;
        this.epochs = epochs;
        
        this.tolerance = 1e-10;
        this.biggerdw = 0;
        this.oldW = [];
        this.oldV = [];

        this.error = 0;
        this.epochError = 0;

        // Passo 0 - peso camada escondida e ultima camada
        this.v = this.randomUniform(this.data.nro_in*this.data.nro_out + this.data.nro_out);
        this.w = this.randomUniform(this.data.nro_in*this.data.nro_out + this.data.nro_out);
        
        this.dw = 0;
        this.dw_b = 0;

        this.dv = 0;
        this.dv_b = 0;

        // saida - camada escondida
        this.zin_ = this.zeros(this.data.nro_out); //nro_camada_escondida*nro_out, derivada
        this.z = this.zeros(this.data.nro_out);

        // saida - ultima camada
        this.yin_ = this.zeros(this.data.nro_out);
        this.y = this.zeros(this.data.nro_out);
    }

    zeros(n) {
        const randomList = [];
        for (let i = 0; i < n; i++) {
            randomList.push(0);
        }
        return randomList;
    }

    randomUniform(n) {
        const randomList = [];
        for (let i = 0; i < n; i++) {
            const nro = Math.random()-0.5; // [-0.5,0.5)
            randomList.push(nro);
        }
        return randomList;
    }

    calculateOut(x,w,y,yin_) {
        for(let j=0; j<this.data.nro_out; j++) { // colunm
            let c = w[w.length-1];
            // console.log(c)
            for(let i=0; i<this.data.nro_in; i++) { // line a1xnro_in
                c += x[i]*w[i*this.data.nro_out+j];
            } 
            // console.log(c)
            yin_[j] = this.activationF_(c);
            y[j] = this.activationF(c); 
        }
        // console.log("Dentro: ", y)
    }

    // Bipolar sigmoid
    activationF(x) {
        return 2/(1+Math.exp(-x)) - 1 
    }
    activationF_(x) { // derivada
        let fx = this.activationF(x);
        return 0.5*(1+fx)*(1-fx) 
    }

    updateWeights() {
        this.error = 0;
        let errYTarget, deltaK, deltainJ = 0, deltaJ;
        
        for(let j=0; j<this.data.nro_out; j++) {
            errYTarget = this.data.t[j]-this.y[j];
            deltaK = errYTarget*this.yin_[j];
            
            for(let i=0; i<this.data.nro_in; i++) {
                deltainJ += deltaK*this.w[i*this.data.nro_out+j];

                // Passo 6
                this.dw = this.alpha*deltaK*this.z[i]; 
                this.dw_b = this.alpha*deltaK;

                // Passo 8
                this.w[i*this.data.nro_out+j] += this.dw;
                this.w[this.w.length-1] += this.dw_b;                
            }
            this.error += errYTarget*errYTarget;
        }
        this.error /= (2*this.data.nro_out)

        for(let j=0; j<this.data.nro_out; j++) {
            deltaJ = deltainJ*this.zin_[j];
            
            for(let i=0; i<this.data.nro_in; i++) {
                // Passo 7
                this.dv = this.alpha*deltaJ*this.data.x[i]; 
                this.dv_b = this.alpha*deltaJ;

                // Passo 8
                this.v[i*this.data.nro_out+j] += this.dv;
                this.v[this.v.length-1] += this.dv_b;               
            }
        }
    }

    validate() {
        for(let i in this.w){ // w é matrix represeentada como vetor
            this.dw = this.oldW[i] - this.w[i];
            this.dw = (this.dw > 0) ? this.dw : -this.dw;
            if(this.biggerdw<this.dw){
                this.biggerdw = this.dw;
            }
        }
        for(let i in this.v){ // w é matrix represeentada como vetor
            this.dv = this.oldV[i] - this.v[i];
            this.dv = (this.dv > 0) ? this.dv : -this.dv;
            if(this.biggerdw<this.dv){
                this.biggerdw = this.dv;
            }
        }
    }

    feedForward({x=this.data.x}){
        // Passo 4
        console.log(x, this.v, this.z, this.zin_)
        this.calculateOut(x, this.v, this.z, this.zin_);
        // console.log("Fora: ", this.z, "\n")

        // Passo 5
        this.calculateOut(this.z, this.w, this.y, this.yin_);        
        // console.log("Fora: ", this.y, "\n")
    }

    backForward(){
        // Passo 6
        this.updateWeights(); 
    }

    train(){
        this.biggerdw = 0;
        this.oldW = this.w.slice(0);
        this.oldV = this.v.slice(0);
        this.epochError = 0;

        // Passo 3
        while(this.data.case<this.data.nro_cases) {
            // console.log(this.data.case)
            this.feedForward({});
            this.backForward();
            
            this.epochError += this.error;
            this.data.case++;
        }
        this.data.case = 0;
        this.validate();
        this.epochError /= this.data.nro_cases;
        
        return this.biggerdw >= this.tolerance;
    }

    predict(x){
        x = Array.isArray(x) ? x : [x];
        this.feedForward({x:x})
        return this.y;
    }
}

// inicio

function multiLayerPerceptron(epochs=1000, alpha=0.01){
    const data = new DataSet({data: x, nro_in: 1,
                            target: y, nro_out: 1});

    let mlp = new MLP({data, epochs: 3, alpha});
    let epoch = 1;
    let continueCondition = true;

    let dotsError = [], dotsWinRate = [], dotsTarget = [], dotsY = [], maior = 0, menor = 100;;

    console.log(`Initial: `)
    console.log(mlp.v)
    console.log(mlp.w)

    // Passo 1
    while ((epoch <= mlp.epochs) && continueCondition) {
        console.log(`Epoch ${epoch}: `)
        continueCondition = mlp.train();
        dotsError.push({x:epoch, y:mlp.epochError})
        dotsWinRate.push({x:epoch, y:mlp.epochError})

        console.log(mlp.v)
        console.log(mlp.w)
        epoch++;
    }

    for(let i in data.target) {
        prediction = mlp.predict(data.data[i])
        dotsTarget.push({x:data.data[i],y:data.target[i]});
        dotsY.push({x:data.data[i],y:prediction});
        
        erro = data.target[i]-prediction;
        erro = erro < 0 ? -erro : erro;
        if(maior < erro){
            maior = erro;
        }
        if(menor > erro){
            menor = erro;
        }
    }
    
    
    showInfo({epoch, mlp, menor, maior, dots:dotsError, dots2:dotsWinRate, dots3:dotsTarget, dots4:dotsY})
}