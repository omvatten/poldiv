import streamlit as st
import pandas as pd
import numpy as np
import pickle
from bokeh.plotting import figure
from bokeh.palettes import Category20
from bokeh.models import ColumnDataSource, HoverTool

header = st.container()
topplistan = st.container()
alphadiv = st.container()
betadiv = st.container()
metoder = st.container()

df = pd.read_csv('data/fixed_data_combined.csv', index_col=0, encoding='latin1')
parties = ['M', 'C', 'L', 'Kd', 'S', 'Mp', 'V', 'Sd', 'Övr']
with open('data/alpha_dis_q1_dict.pickle', 'rb') as f:
    q1 = pickle.load(f)
with open('data/alpha_dis_q2_dict.pickle', 'rb') as f:
    q2 = pickle.load(f)

with header:
    st.title('Politisk mångfald i Sverige')
    st.markdown("""
                - Hur har mångfalden av politiska partier förändrats över tid?
                - Vilka regioner har högst respektive lägst mångfald?
                - Hur skiljer sig röstfördelningen mellan olika regioner?
                """)
    st.markdown('Hög politisk mångfald innebär att många olika partier får röster. '
                'Om t.ex. 5 olika partier får 20% var av rösterna så är mångfalden högre än om bara 2 partier får 50% var.')
    st.markdown('Jag har använt index från ekologi för att beräkna både mångfald och skillnad i röstfördelning mellan olika regioner i Sverige. '
                'Om du är intresserad av hur dessa beräknas kan du titta längst ner på sidan.')
    q = st.radio(label='Mångfaldsindex (mer detaljer om detta längst ner för den som är intresserad)', options=[1, 2], horizontal=True)

if q == 1:
    data = q1
elif q == 2:
    data = q2

with topplistan:
    st.header('Topplistan')
    st.markdown('Vi börjar med topplistan. Här ser vi de 10 kommuner med högst- och lägst mångfald. '
                'Vi ser även de 10 kommuner som har en röstfördelning som är mest lik eller olik Sveriges som helhet')
    tl1, tl2 = st.columns([1, 5])
    with tl1:
        tlyr = st.selectbox('Välj år för topplistan', sorted(df['Year'].unique().tolist())[::-1])
    with tl2:
        alpha = data[tlyr][0][data[tlyr][0]['Kat'] == 'Kommun']
        alphaTop10 = alpha.sort_values('alpha', ascending=False).index[:10]
        alphaBot10 = alpha.sort_values('alpha', ascending=False).index[-10:][::-1]

        dis = data[tlyr][1]
        beta = dis.loc[alpha.index, 'Totalt för riket'].sort_values(ascending=False)
        betaTop10 = beta.index[:10]
        betaBot10 = beta.index[-10:][::-1]

        df_TL = pd.DataFrame(index=range(1, 11), columns=['Högst mångfald', 'Lägst mångfald', 'Lik riket', 'Olik riket'])
        df_TL['Högst mångfald'] = alphaTop10
        df_TL['Lägst mångfald'] = alphaBot10
        df_TL['Lik riket'] = betaBot10
        df_TL['Olik riket'] = betaTop10

        st.table(df_TL)
    
with alphadiv:
    #Tidsgraf
    st.header('Förändring i mångfald över tid')
    totl = df.loc[df['Kat'] == 'Totalt', 'Region'].unique().tolist()
    koml = sorted(df.loc[df['Kat'] == 'Kommun', 'Region'].unique().tolist())
    kretsl = sorted(df.loc[df['Kat'] == 'Valkrets', 'Region'].unique().tolist())
    plotregs = st.multiselect(label='Lägg till eller ta bort regioner i diagrammet', options=totl+koml+kretsl, default='Totalt för riket')
    
    alpha_time = pd.DataFrame(pd.NA, index=df['Year'].unique(), columns=df['Region'].unique())
    for y in df['Year'].unique():
        alpha_time.loc[y, data[y][0].index] = data[y][0]['alpha'].to_numpy()
    atPlot = alpha_time[plotregs] 

    fig1 = figure(plot_height=300, plot_width=800, x_axis_label='År', y_axis_label='Mångfald', y_range=(0.9, 9.1))
    for i, c in enumerate(atPlot.columns):
        fig1.line(atPlot.index, atPlot[c], legend_label=c, line_width=2, color=Category20[12][i])
        fig1.circle(atPlot.index, atPlot[c], size=8, color=Category20[12][i])
    fig1.add_layout(fig1.legend[0], 'right')
    st.bokeh_chart(fig1)     

    #Ranking
    st.header('Mångfald i olika regioner')
    st.markdown('Välj vilka valår du vill undersöka och om du vill titta på kommuner eller valkretsar eller båda.')
    st.markdown('Genom att hålla muspekaren över en punkt i diagrammet får du information om regionen som punkten avser. Du kan även zoom in på delar av diagrammet. '
                'Den horizontella linjen är mångfalden för Sverige som helhet.')
    ar1, ar2 = st.columns([2,1])
    with ar1:
        plotyr = st.multiselect('Valår', sorted(df['Year'].unique().tolist())[::-1], default=2018)

    with ar2:
        adiv_kat = st.multiselect('Regioner', ['Kommuner', 'Valkretsar'], default='Kommuner')
    plotkat = [c[:-2] for c in adiv_kat]
        
    arPlot = {}
    for y in plotyr:
        alpha_all = data[y][0].sort_values('alpha', ascending=False)
        alpha_kat = alpha_all[alpha_all['Kat'].isin(plotkat)].copy()
        alpha_kat['Rank'] = np.arange(len(alpha_kat))+1
        alpha_kat['Riket'] = alpha_all.loc['Totalt för riket', 'alpha']
        arPlot[y] = ColumnDataSource(alpha_kat)

    fig2 = figure(plot_height=300, plot_width=800, x_axis_label='Mångfaldsranking', y_axis_label='Mångfald', y_range=(0.9, 9.1))
    for i, y in enumerate(plotyr):
        fig2.line(x='Rank', y='Riket', source=arPlot[y], line_width=2, color=Category20[12][i])
        fig2.circle(x='Rank', y='alpha', source=arPlot[y], legend_label=str(y), size=5, color=Category20[12][i], selection_alpha=0.5)

    tooltips = [('Region', '@Region'), ('Antal röstande', '@Votes'), ('Mångfaldsranking', '@Rank')]
    fig2.add_tools(HoverTool(tooltips=tooltips))
    fig2.add_layout(fig2.legend[0], 'right')
    st.bokeh_chart(fig2)     

with betadiv:
    st.header('Skillnad i röstfördelning mellan regioner')
    st.markdown('Välj vilket valår du vill undersöka och om du vill titta på kommuner eller valkretsar eller båda. '
                'Tabellerna visar ett urval av parvisa jämförelser mellan regioner. ')

    b1, b2 = st.columns([1, 5])
    with b1:
        b1yr = st.selectbox('Välj år för jämförelsen', sorted(df['Year'].unique().tolist())[::-1])
    with b2:
        betaregs = st.multiselect(label='Regioner att jämföra', options=['Kommuner', 'Valkretsar'], default='Kommuner')
        betaregs = [c[:-2] for c in betaregs]
        keeps = data[b1yr][0][data[b1yr][0]['Kat'].isin(betaregs)].index
        dis = data[b1yr][1].loc[keeps, keeps]
        dis.index.name = None
        dis.columns.name = None
        dis = dis.where(np.triu(np.ones(dis.shape), k=1).astype(bool)).stack().reset_index()
        dis.columns = ['Region 1', 'Region 2', 'Skillnad (%)']
        dis['Skillnad (%)'] = dis['Skillnad (%)']*100
        dis = dis.sort_values('Skillnad (%)', ascending=False)
        table1 = dis.groupby('Region 1').first().sort_values('Skillnad (%)', ascending=False).iloc[:5]
        table2 = dis.groupby('Region 1').last().sort_values('Skillnad (%)', ascending=True).iloc[:5]

        table1['Skillnad (%)'] = table1['Skillnad (%)'].round(1).astype(str)
        table2['Skillnad (%)'] = table2['Skillnad (%)'].round(1).astype(str)
        table1.index.name = 'Region 1'; table1.reset_index(inplace=True); table1.index = range(1, len(table1)+1)
        table1.index.name = 'Region 1'; table2.reset_index(inplace=True); table2.index = range(1, len(table2)+1)

        st.markdown('Fem par med stor skillnad i röstfördelning:')
        st.table(table1)
        st.markdown('Fem par med väldigt liknande röstfördelning:')
        st.table(table2)

with metoder:
    st.title('Metoder')
    st.subheader('Hur mäter vi mångfald?')
    st.markdown('Tänk att vi har 5 partier. Varje parti får 20% av rösterna. Här är det logiskt att säga att den politiska mångfalden = 5. '
                'Om ett enda parti istället fick 100% av rösterna så är det logiskt att den politiska mångfalden = 1.')
    m1, m2 = st.columns([3, 1])
    with m1:
        st.markdown('Om fördelningen ser ut på något annat sätt, t.ex. som i tabellen till höger, '
                    'så borde mångfalden ligga någonstans mellan 1 och 5.')
        st.markdown('Det finns olika sätt att beräkna mångfald. Titta på tabellen till höger: ska mångfalden vara 5 eftersom fem partier får röster '
                    'eller ska den kanske vara 3 eftersom tre partier för mycket fler röster än de andra två? ' 
                    'Jag har utgått från något som kallas Hill nummer, som ofta används inom ekologi för att mäta mångfalden av arter. '
                    'Högst upp på sidan kan man välja mångfaldsindex 1 eller 2. '
                    'Det första, 1, betyder att varje parti viktas exakt enligt sin fraktion av rösterna. '
                    'Det andra, 2, innebär att man tar fraktionen i kvadrat, vilket betyder att stora partier viktas mer än sin procentsats i jämförelse med små partier. '
                    'Dessa index används också inom *Political Science* för att beräkna *effective number of parties*, se t.ex. '
                    'Laakso and Taagepera (1979), *Comparative Political Studies*, 12(1), 3-27. '
                    'Eller kolla Wikipedia: https://en.wikipedia.org/wiki/Effective_number_of_parties')
    with m2:
        tab = pd.DataFrame([40, 30, 20, 5, 5], columns=['Röster (%)'], index=['Parti 1', 'Parti 2', 'Parti 3', 'Parti 4', 'Parti 5'])
        tab.index.name = 'Parti'
        st.table(tab)
        
    st.subheader('Hur mäter vi skillnad i röstfördelning?')
    st.markdown('Jag har använt samma familj av beräkningsmetoder som beskrivs för mångfalden ovan, för att beräkna även skillnad i röstfördelning mellan olika regioner. '
                'Det betyder att index 1 viktar partierna enligt deras fraktion av rösterna medan index 2 tar fraktionen i kvadrat. '
                'Det betyder alltså att skillnader i fördelningen till de stora partierna blir viktigare för index 2. '
                'Ett sätt att tolka de procentuella skillnader mellan regioner som visas i tabellerna ovan är '
                'att de visar andelen partisympatier i region 1 som inte delas med region 2.'
                )
    st.subheader('Data')
    st.markdown('Jag laddade ner datan från SCB via deras API (https://www.scb.se/en/services/open-data-api/api-for-the-statistical-database/).' 
                '  pyscbwrapper (https://github.com/kirajcg/pyscbwrapper) förenklade detta för mig. '
                'Jag har inkluderat röster till de åtta riksdagspartierna. Övriga småpartier är ihopbakade till ett "nionde" parti.')
    st.markdown('Jag som har gjort sidan heter Oskar Modin. Kan nås på omvatten|AT|gmail.com')
