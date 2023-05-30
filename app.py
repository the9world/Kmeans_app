import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
# 표준화 = StandardScaler, 정규화 = MinMaxScaler 
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("K-Means 클러스터링 앱")
    # 1. csv 파일을 업로드 할 수 있다.
    csv_file = st.file_uploader('CSV 파일 업로드', type=['csv'] )
    
    if csv_file is not None:
        # 업로드한 csv 파일을 데이터 프레임으로 읽고
        df= pd.read_csv(csv_file)
        
        st.dataframe(df)
        
        st.subheader('NaN 데이터 확인')
        
        st.dataframe( df.isna().sum() )
    
        st.subheader('결측 값 처리한 결과')
        df = df.dropna()
        df.reset_index(inplace=True, drop=True)
        st.dataframe(df)
        
        st.subheader('클러스터링에 사용할 컬럼 선택')
        selcted_columns= st.multiselect('X로 사용할 컬럼을 선택하세요.', df.columns)
        
        if len(selcted_columns) != 0 :
            X= df[selcted_columns]
            st.dataframe(X)

            # 숫자로 된 새로운 데이터프레임을 만든다.
            X_new = pd.DataFrame()

            for name in X.columns :
            # print(name)

                # 데이터가 문자열이면, 데이터의 종류가 몇개인지 확인한다.
                if X[ name ].dtype == object :

                    if X[name].nunique() >= 3 :
                        # 원핫 인코딩한다.
                        ct = ColumnTransformer( [('encoder', OneHotEncoder() ,[0] )] , 
                                        remainder= 'passthrough' )
                        
                        col_names = sorted( X[name].unique() )

                        X_new[col_names] = ct.fit_transform(  X[name].to_frame()  )

                    else :
                        # 레이블 인코딩 한다.
                        label_encoder = LabelEncoder()
                        X_new[name] = label_encoder.fit_transform( X[name] )
                
                # 숫자 데이터일때의 처리는 여기서
                else :
                    X_new[name] = X[name]

            st.subheader('문자열을 숫자로 바꿔줍니다.')
            st.dataframe(X_new)
            
            # 피쳐 스케일링 한다.(MinMaxScaler 사용)
            st.subheader('피쳐 스케일링을 합니다.')
            scaler = MinMaxScaler()
            X_new= scaler.fit_transform(X_new)
            st.dataframe(X_new)
            
            # 유저가 입력한 파일의 데이터 갯수를 세어서
            # 해당 데이터의 갯수가 10보다 작으면, 
            # 데이터의 갯수까지만 wcss 를 구하고, 10보다 크면 10개로
            # 데이터 갯수는 인덱스!
            
            # X_new.shape[0] # 행만 세어줌 8
            if X_new.shape[0] < 10: # [0]데이터 억세스 기호 shape (8, 6) 출력 값은 튜플
                # 이 갯수 만큼만 밑에 for문에 레인지?
                max_count = X_new.shape[0]
            else: 
                max_count = 10 
            
            wcss = [] # 비어 있는 list를 만들어 놓고 추후 wcss값을 받을 것이다.
            for k in range(1, max_count+1) : 
                kmeans= KMeans(n_clusters= k, random_state= 5,
                               n_init='auto') # n_init은 에러 안뜨게 하는 것
                kmeans.fit(X_new) # 학습
                wcss.append(kmeans.inertia_) # wcss값 가져와 append로 추가

            x = np.arange(1, max_count+1)
            
            fig = plt.figure(figsize=(10,5))
            plt.plot(x, wcss) # .plot : 표 만듬,  x축은 클러스터의 갯수,  y축은 wcss ,
            plt.title('The Elbow Method') # 표 제목
            plt.xlabel('Number of Clusters') # x축 이름
            plt.xlabel('WCSS') # y축 이름
            st.pyplot(fig)
            
            st.subheader('클러스터링 갯수 선택')
            
            k= st.number_input('K를 선택', 1, max_count, value=3) # 밸류: 초기 값 
            # ↓ 정답은 없으니 위의 표를 구하고 내가 몇개로 할지 정하는 것.
            kmeans= KMeans(n_clusters= k, random_state= 5, n_init='auto')
            
            y_pred = kmeans.fit_predict(X_new) # 이거 설명을 못 들었음 나중에 찾아보자
            df['Group'] = y_pred
            
            st.subheader('그루핑 정보 표시')
            st.dataframe(df)
            
            st.subheader('보고 싶은 그룹을 선택!')
            group_number= st.number_input('그룹번호 선택', 0, k-1)
            
            st.dataframe(df.loc[df['Group']== group_number, ] )
            
            # df를 csv 파일로 저장
            df.to_csv('result.csv', index=False, encoding="utf-8-sig")
            # index 열은 제거하고 저장, utf-8-sig: 한글이 깨지지 않는 utf-8

if __name__=='__main__':
    main()
