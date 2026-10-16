#include <math.h>
/* Same continuous-time CTH/IDM laws as the Python reference; fixed maximum
   RK4 step makes observation-rate comparisons independent of solver step. */
static double accel(int law, const double *p, double s, double v, double u){
    if(law>=2){
        int dict=law-2,n=4;double d=u-v,F[27]={1,s,v,d};
        if(dict==1){n=10;F[4]=s*s;F[5]=v*v;F[6]=d*d;F[7]=s*v;F[8]=s*d;F[9]=v*d;}
        if(dict==2){double e=(s-3-1.5*v)/10;n=9;F[4]=e*e;F[5]=e*d;F[6]=d*d;F[7]=tanh(d);F[8]=e*fabs(e);}
        if(dict==3||dict==4){double si=1/(s*s),vv=v/30;n=11;F[4]=vv*vv*vv*vv;F[5]=si;F[6]=v*si;F[7]=v*v*si;F[8]=v*d*si;F[9]=v*v*d*si;F[10]=v*v*d*d*si;
            if(dict==4){double e=(s-3-1.5*v)/10;n=14;F[11]=e*e;F[12]=e*d;F[13]=tanh(d);}}
        if(dict==5){double ss=(s-p[6])/p[9],vs=(v-p[7])/p[10],us=(u-p[8])/p[11];int q=0;n=27;
            double sp[3]={1,ss,ss*ss},vp[3]={1,vs,vs*vs},up[3]={1,us,us*us};
            for(int k=0;k<3;k++)for(int j=0;j<3;j++)for(int i=0;i<3;i++)F[q++]=sp[i]*vp[j]*up[k];}
        double a=0;for(int j=0;j<n;j++)a+=p[12+j]*(F[j]-p[12+n+j])/p[12+2*n+j];
        if(p[0]>=0)a+=accel((int)p[0],p+1,s,v,u);return a;
    }
    if(law==0) return p[0]*(s-p[2]-p[1]*v)+p[3]*(u-v);
    double star=p[3]+fmax(0.,v*p[2]+v*(v-u)/(2*sqrt(p[0]*p[1])));
    double q=v/p[4], r=star/s;
    return p[0]*(1-q*q*q*q-r*r);
}
int replay(int law, const double *p, int n, const double *t,
           const double *u, double s, double v, double maxdt, double *out){
    out[0]=s;out[1]=v;
    for(int i=1;i<n;i++){
        double span=t[i]-t[i-1]; int steps=(int)ceil(span/maxdt-1e-9);
        if(steps<1)steps=1; double h=span/steps;
        for(int j=0;j<steps;j++){
            double u0=u[i-1]+(u[i]-u[i-1])*j/steps;
            double u1=u[i-1]+(u[i]-u[i-1])*(j+1)/steps,um=(u0+u1)/2;
            double as=u0-v,av=accel(law,p,s,v,u0);
            double bs=um-v-h*av/2,bv=accel(law,p,s+h*as/2,v+h*av/2,um);
            double cs=um-v-h*bv/2,cv=accel(law,p,s+h*bs/2,v+h*bv/2,um);
            double ds=u1-v-h*cv,dv=accel(law,p,s+h*cs,v+h*cv,u1);
            s+=h*(as+2*bs+2*cs+ds)/6;v+=h*(av+2*bv+2*cv+dv)/6;
            if(!isfinite(s)||!isfinite(v)||fabs(s)>1e4||fabs(v)>1e3){
                for(int k=i;k<n;k++){out[2*k]=NAN;out[2*k+1]=NAN;}return 1;
            }
        }
        out[2*i]=s;out[2*i+1]=v;
    }
    return 0;
}
