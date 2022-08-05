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

totl = df.loc[df['Kat'] == 'Totalt', 'Region'].unique().tolist()
koml = sorted(df.loc[df['Kat'] == 'Kommun', 'Region'].unique().tolist())
kretsl = sorted(df.loc[df['Kat'] == 'Valkrets', 'Region'].unique().tolist())
allregs = totl + koml + kretsl

with header:
    st.title('Politisk mångfald i Sverige')
    st.markdown("""
                - Hur har mångfalden av politiska partier förändrats över tid?
                - Vilka regioner har högst respektive lägst mångfald?
                - Hur skiljer sig röstfördelningen mellan olika regioner och hur har skillnaden förändrats över tid?
                """)
    st.markdown('Hög politisk mångfald innebär att många olika partier får röster. '
                'Om t.ex. 5 olika partier får 20% var av rösterna så är mångfalden högre än om bara 2 partier får 50% var.')
    st.markdown('Jag har använt index från ekologi för att beräkna både mångfald och skillnad i röstfördelning mellan olika regioner i valet till riksdagen i Sverige. '
                'Om du är intresserad av hur dessa beräknas så kan du titta längst ner på sidan.')
    q = st.radio(label='Mångfaldsindex (mer detaljer om detta längst ner).', options=[1, 2], horizontal=True)

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
    plotregs = st.multiselect(label='Lägg till eller ta bort regioner i diagrammet', options=allregs, default='Totalt för riket')
    
    alpha_time = pd.DataFrame(pd.NA, index=df['Year'].unique(), columns=df['Region'].unique())
    for y in df['Year'].unique():
        alpha_time.loc[y, data[y][0].index] = data[y][0]['alpha'].to_numpy()
    atPlot = alpha_time[plotregs] 

    fig1 = figure(plot_height=300, plot_width=800, x_axis_label='År', y_axis_label='Mångfald', y_range=(0.8*min(atPlot.min()), 1.2*max(atPlot.max())))
    for i, c in enumerate(atPlot.columns):
        fig1.line(atPlot.index, atPlot[c], legend_label=c, line_width=2, color=Category20[12][i])
        fig1.circle(atPlot.index, atPlot[c], size=8, color=Category20[12][i])
    fig1.add_layout(fig1.legend[0], 'right')
    st.bokeh_chart(fig1)     

    #Ranking
    st.header('Mångfald i olika regioner')
    st.markdown('Välj vilka valår du vill undersöka och om du vill titta på kommuner eller valkretsar eller båda.')
    st.markdown('Genom att hålla muspekaren över en punkt i diagrammet får du information om regionen som punkten avser. Du kan även zooma in på delar av diagrammet. '
                'Den horisontella linjen är mångfalden för Sverige som helhet.')
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
    st.markdown('Välj två regioner och se hur skillnaden i röstfördelning mellan dem har förändrats över tid.')

    #Tidsgraf
    b1, b2 = st.columns([1, 1])
    with b1:
        reg1 = st.selectbox(label='Välj region 1', options=allregs, index=199)
    with b2:
        reg2 = st.selectbox(label='Välj region 2', options=allregs, index=58)
    
    beta_time = pd.DataFrame(pd.NA, index=df['Year'].unique(), columns=['dis'])
    for y in df['Year'].unique():
        if reg1 in data[y][1].index and reg2 in data[y][1].index:
            beta_time.loc[y, 'dis']  = 100*data[y][1].loc[reg1, reg2]

    figB = figure(title=reg1+' vs '+reg2, plot_height=300, plot_width=800, x_axis_label='År', y_axis_label='Skillnad (%)', y_range=(beta_time['dis'].min()*0.8, beta_time['dis'].max()*1.2))
    figB.line(beta_time.index, beta_time['dis'], line_width=2, color=Category20[12][0])
    figB.circle(beta_time.index, beta_time['dis'], size=8, color=Category20[12][0])
    st.bokeh_chart(figB)     

    st.markdown('Vi kan också titta på de parvisa jämförelser av regioner som har högst och lägst skillnad i röstfördelning. '
                'Välj valår och om du vill titta på kommuner eller valkretsar, och se resultatet i tabellerna nedan.')
    b3, b4 = st.columns([2, 8])
    with b3:
        b1yr = st.selectbox('Välj år', sorted(df['Year'].unique().tolist())[::-1])
    with b4:
        betaregs = st.multiselect(label='Kommuner, valkretsar eller båda?', options=['Kommuner', 'Valkretsar'], default='Kommuner')

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
                    'Jag har utgått från något som kallas Hill-nummer, som ofta används inom ekologi för att mäta mångfalden av arter. '
                    'Högst upp på sidan kan man välja mångfaldsindex 1 eller 2. '
                    'Det första, 1, betyder att varje parti viktas exakt enligt sin fraktion av rösterna. '
                    'Det andra, 2, innebär att man tar fraktionen i kvadrat, vilket betyder att stora partier viktas mer än sin procentsats i jämförelse med små partier. '
                    'Index 2 kommer därför ge lägre värden än index 1 eftersom 2an i högre grad är ett mått på "antalet stora partier". '
                    'Dessa index används också inom *Political Science* för att beräkna *effective number of parties*, se t.ex. '
                    'Laakso and Taagepera (1979), *Comparative Political Studies*, 12(1), 3-27. '
                    'Eller kolla Wikipedia: https://en.wikipedia.org/wiki/Effective_number_of_parties')
    with m2:
        tab = pd.DataFrame([40, 30, 20, 5, 5], columns=['Röster (%)'], index=['Parti 1', 'Parti 2', 'Parti 3', 'Parti 4', 'Parti 5'])
        tab.index.name = 'Parti'
        st.table(tab)
        
    st.subheader('Hur mäter vi skillnad i röstfördelning?')
    st.markdown('För att beräkna skillnad i röstfördelning mellan olika regioner har jag använt samma familj av beräkningsmetoder som beskrivs för mångfalden ovan. '
                'Det betyder att index 1 viktar partierna enligt deras fraktion av rösterna medan index 2 tar fraktionen i kvadrat. '
                'Det betyder alltså att skillnader i fördelningen till de stora partierna blir viktigare för index 2. '
                'Ett sätt att tolka de procentuella skillnader mellan regioner som visas i tabellerna ovan är kanske '
                'att de visar andelen partisympatier i region 1 som inte delas med region 2. '
                'Tidigare har jag använt de här metoderna för att beräkna skillanden i sammansättningen av mikroorganismer i olika miljöer. '
                'Kolla t.ex. Modin et al. (2020), *Microbiome* 132 (https://doi.org/10.1186/s40168-020-00909-7)')
    st.subheader('Data')
    st.markdown('Jag laddade ner datan från SCB via deras API (https://www.scb.se/en/services/open-data-api/api-for-the-statistical-database/).' 
                '  pyscbwrapper (https://github.com/kirajcg/pyscbwrapper) förenklade detta för mig. '
                'Jag har inkluderat röster till de åtta riksdagspartierna. Övriga småpartier är ihopbakade till ett "nionde" parti.')
    st.markdown('Jag som har gjort sidan heter Oskar Modin. Kan nås på omvatten|AT|gmail.com')
