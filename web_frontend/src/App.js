import './App.css';
import addBtn from './assets/image/add.png';
import msgIcon from './assets/image/message.svg';
import share from './assets/image/share.svg';
import sendBtn from './assets/image/send.svg';
import userIcon from './assets/image/user.png';
import sidebar from './assets/image/sidebar.png';
import logo from './assets/image/logo.png';
import {useEffect, useRef, useState} from "react";
import axios from 'axios';
import Plot from 'react-plotly.js';

function App() {
    // User Input
    const paperLink = 'https://www.jonathanfan.site/';
    const [inputString, setInputString] = useState('');
    const [messages, setMessages] = useState([
            {
                text: <div>
                    <p>
                        Hi, I am FenUI. To generate an attention index, please send a message that clearly specifies a query in any format. You can also specify a start date, end date, and p-value. If not, I will use default values 1984-01-01, 2021-12-31, and 0.01. Please note, the longer your desired time frame, the longer it will take to generate.
                    </p>
                    <p>
                        Check out our paper <a href={paperLink} style={{ color: '#72bdd4', textDecoration: 'none' }} target="_blank">here</a> for more info. Happy generating!
                    </p>
                </div>,
                isBot: true,
            }
        ]);

    // Auto Scroll
    const msgEnd = useRef(null);
    useEffect(() => {
        msgEnd.current.scrollIntoView({ behavior: 'auto' });
    }, [messages]);

    // Simulate Typing with Callback
    const simulateTyping = (response, isQuery, callback, onComplete) => {
        let initialDelay = isQuery ? 300 : 600;
        let delay = isQuery ? 100 : 200;

        const words = response.split(" ");
        let currentText = "";
        let cumulativeDelay = initialDelay;

        words.forEach((word, index) => {
            cumulativeDelay += delay + Math.random() * delay;

            setTimeout(() => {
                currentText += word + " ";
                setMessages(prevMessages => {
                    // Update only the last message (which should be the bot typing animation)
                    let newMessages = [...prevMessages];
                    newMessages[newMessages.length - 1] = { text: currentText.trim(), isBot: true };
                    return newMessages;
                });

                // Execute callback after the last word
                if (index === words.length - 1) {
                    if (callback) {
                        setTimeout(callback, delay);
                    }
                    // Add this check here
                    if (onComplete) {
                        setTimeout(onComplete, 0);
                    }
                }
            }, cumulativeDelay);
        });
    };

    // Send Input to Backend and Handle Bot is Typing
    const [csvDataUrl, setCsvDataUrl] = useState(null);
    const [queryDataUrl, setCsvQueryUrl] = useState(null);
    const [botIsTyping, setBotIsTyping] = useState(false);
    const handleSend = async () => {
        if (botIsTyping) {
            // Bot is still typing, do not allow user input
            return;
        }

        const userInput = inputString;
        setInputString('');

        // Add user's message and an initial bot message
        setMessages(prevMessages => [
            ...prevMessages,
            { text: userInput, isBot: false },
            { text: "Bot is typing...", isBot: true }
        ]);

        setBotIsTyping(true);

        try {
            // Call to Flask API Worked
            // Local
            const response = await axios.post('http://localhost:5000/generate_plot', { "input_str": userInput });

            // // Prod (GCP)
            // const response = await axios.post('https://webbackend-yhzjtissga-ue.a.run.app/generate_plot', { "input_str": userInput });

            // // Prod (Ngrok)
            // const response = await axios.post("https://c4f3-128-255-234-12.ngrok-free.app/generate_plot", { "input_str": userInput });

            // Parse the plot JSON data
            const plotData = JSON.parse(response.data.gen_plot);

            // Get the generated index data
            const indexCsvData = response.data.gen_index;

            // Get the expanded query data
            const expandQueryData = response.data.expand_query

            // Create a Blob from the CSV data and create an object URL for it
            const blob = new Blob([indexCsvData], { type: 'text/csv' });
            const dataUrl = window.URL.createObjectURL(blob);
            setCsvDataUrl(dataUrl);

            const blob_query = new Blob([expandQueryData], { type: 'text/csv' });
            const queryUrl = window.URL.createObjectURL(blob_query);
            setCsvQueryUrl(queryUrl);

            // Retrieve generated combination data
            const combineCsvData = response.data.gen_combine;
            const parsedCombineData = parseCsvData(combineCsvData);
            setPlotDataDetails(parsedCombineData);

             // Simulate typing and then update the message with the plot
            simulateTyping("Here is your generated attention index, click on any date to view the article that is most related toward your desired label! ",
                false,
                () => {
                setMessages(prevMessages => [
                    ...prevMessages.slice(0, -1),
                    {
                        type: "plot",
                        content: plotData,
                        isBot: true,
                        text: "Here is your generated attention index, click on any date to view the article that is most related toward your desired label! ",
                    }
                ]);
            }, () => setBotIsTyping(false));
        } catch (error) {
            console.error("Error caught: ", error);
            if (error.response && error.response.data && error.response.data.error) {
                // Handle error response (due to error in input format, date, or transform)
                simulateTyping(error.response.data.error, false, null, () => setBotIsTyping(false));
            } else if (axios.isAxiosError(error)) {
                // The error is related to Axios or the API call
                simulateTyping("An error occurred while processing your request, please try again.", false, null, () => setBotIsTyping(false));
            } else {
                // The error is something else (maybe a bug in the code)
                simulateTyping("An unexpected error occurred, please try again.", false, null, () => setBotIsTyping(false));
            }
        }
    };

    // Handle Query
    const handleQuery = async (e) => {
        const text = e.target.value
        let res = "";
        setBotIsTyping(true);

        if (text === "About Us") {
            res = "This service was developed by Jonathan Fan, Yinan Su, and Leland Bybee who are active researchers at Yale University, John Hopkins University, and University of Chicago. We developed this website in hopes " +
                  "of allowing finance researchers to have access to this incredible tool. Currently, the Economic Policy Uncertainty (EPU) Index is a widely-research index that accurately measures uncertainty in the market" +
                  " with regards to all aspects of economics (i.e., FED Policy, New's Articles, etc.). In our research, we introduce a novel method to generate attention indexes that are specifically catered " +
                  "toward a user's interest. For example, our method can generate an attention index that specifically tracks Artificial Intelligence over the past decade. We have also statistically proven, that our own " +
                  " generated 'EPU' index is highly correlated to the actual EPU index, further validating our method. Check out our paper to read more about our methodology! With this tool, " +
                  "you can generate any attention index catered for your interest! " +
                  "If you have any questions regarding data, methodology, and purpose feel free to contact us at either jonathan.fan@yale.edu, ys@jhu.edu, or leland.bybee@yale.edu. We would love to chat!"
        } else if (text === "How to use?") {
            res = "To generate an index, our method requires one parameter: query. The query can be anything, ranging from 'I like playing basketball' to " +
                  "'Uncertainty in the stock price market'. Next there are three optional parameters to specify: start date, end date, and p-value. For start and end date, we currently only provide data generation from 1984-01-01 to 2021-12-31 (default values for start date and end date)." +
                  " Please specify your time frame within this range and make sure the start date " +
                  " comes before end date. Lastly, the p-value (default value is 0.01) determines the significance level used to calculate the threshold for our relu transformation applied on a daily interval (more info in our paper). The basis intuition for this transformation is this: for each date we have an average of around 200 articles. " +
                  " Instead of just averaging the cosine similarity score of these articles, our percentile relu transformations only considers the most relevant articles - effectively removing any noise. That's it!";
        } else if (text === "How does it work?") {
            res = "For data, we first compiled around 900,000 Wall Street Journal articles on a daily timeframe from 1984 to 2021. Each day has around 200 articles. From here, we then " +
                  "used a LLM to generated an embeddings dataset. Now to actually generated the index, we first expand you query using ChatGPT to capture all points of you query. We then retrieve the embeddings of this expanded query and calculate the cosine " +
                  " similarity score of each article's embedding compared with the label embedding. From here, we then apply a p-value relu transformation on each score to eliminate irrelevant articles and calculate the " +
                  " average score per date. Lastly, we then aggregate the daily timeseries to our final monthly timeseries of scores, which accurately represents the attention index with regards to your inputted query. For more information on" +
                  " our methodology and it's validity, check out our paper!"
        }

        // Add user's message and an initial bot message
        setMessages(prevMessages => [
            ...prevMessages,
            { text: text, isBot: false },
            { text: "Bot is typing...", isBot: true }
        ]);
        simulateTyping(res, true, null, () => setBotIsTyping(false));

        setSidebarVisible(false);
    }

    // Handle Enter key press in input
    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    }

    // Handle Plot Click Article Pop-Up
    const [plotDataDetails, setPlotDataDetails] = useState(null);
    const [showPopup, setShowPopup] = useState(false);
    const [selectedData, setSelectedData] = useState(null);

    // Helper function to parse CSV data into JSON
    const parseCsvData = (csvData) => {
        const lines = csvData.trim().split('\n');
        const result = [];

        // Skip the first line (header)
        for (let i = 1; i < lines.length; i++) {
            const currentline = lines[i].split(',');

            // Assuming the order is date, transform_cosine_sim, headline, document
            const date = currentline[0].trim();
            const transformCosineSim = currentline[1].trim();
            const headline = currentline[2].trim();
            const document = currentline.slice(3).join(',').trim();

            result.push({ date, transformCosineSim, headline, document });
        }

        return result;
    };

    // Handle Share
    const handleShare = () => {
    const url = window.location.href;
        navigator.clipboard.writeText(url).then(() => {
            alert("URL copied to clipboard!");
        }).catch(err => {
            console.error('Failed to copy URL: ', err);
        });
    };

    // Handle Expanding Input
    const inputRef = useRef(null);
    useEffect(() => {
        const input = inputRef.current;
        input.style.height = 'auto';
        input.style.height = `${input.scrollHeight}px`;
    });

    // Toggle Sidebar for phones
    const [sidebarVisible, setSidebarVisible] = useState(false);
    const toggleSidebar = () => {
            setSidebarVisible(!sidebarVisible);
    };

    // Web Design
    return (
        <div className="App">
            <button className={`toggle-btn ${sidebarVisible ? 'active' : ''}`} onClick={toggleSidebar}><img src={sidebar} alt="Side Bar" /></button>
            <div className={`sideBar ${sidebarVisible ? 'active' : ''}`}>
                <div className="upperSide">
                    <div className="upperSideTop"><img src={logo} alt="Logo" className="logo" /><span className="brand">FenUI</span></div>
                    <button className="midBtn" onClick={()=>{window.location.reload()}}><img src={addBtn} alt="New Chat" className="addBtn" />New Chat</button>
                    <div className="upperSideBottom">
                        <button className="query" onClick={botIsTyping ? null : handleQuery} value={'About Us'} disabled={botIsTyping}><img src={msgIcon} alt="Query" />About Us</button>
                        <button className="query" onClick={botIsTyping ? null : handleQuery} value={'How to use?'} disabled={botIsTyping}><img src={msgIcon} alt="Query" />How to use?</button>
                        <button className="query" onClick={botIsTyping ? null : handleQuery} value={'How does it work?'} disabled={botIsTyping}><img src={msgIcon} alt="Query" />How does it work?</button>
                    </div>
                </div>
                <div className="lowerSide">
                    <div className="listItems">
                        <img onClick={handleShare} src={share} alt="Share" className="listItemsImg" />Share
                    </div>
                </div>
            </div>
            <div className={`main ${sidebarVisible ? 'active' : ''}`}>
                <div className="chats">
                    {messages.map((message, i)=>
                        <div key={i} className={message.isBot?"chat bot":"chat"}>
                            <img className="chatImg" src={message.isBot ? logo : userIcon} alt="" />
                            <div className="message-text">
                                <div className="message-label">{message.isBot ? "FenUI" : "You"}</div>
                                {message.text === "Bot is typing..."
                                    ? <div className="loading-circle"></div>
                                    : <p className="message-inp">{message.text}</p>
                                }
                                {message.type === "plot" && (
                                    <div className="plot-container">
                                        <Plot
                                            data={message.content.data}
                                            layout={{
                                                ...message.content.layout,
                                                autosize: true,
                                                responsive: true
                                            }}
                                            useResizeHandler={true}
                                            onClick={(event) => {
                                                const date = event.points[0].x;
                                                const details = plotDataDetails.find(detail => detail.date === date);
                                                setSelectedData(details);
                                                setShowPopup(true);
                                            }}
                                        />
                                        {csvDataUrl && (
                                            <a href={csvDataUrl} download="attention_index.csv" className="download-csv-link" style={{ color: 'rgb(120, 180, 240)', textDecoration: 'none', display: 'inline-block', marginTop: '1rem' }}>
                                                Download Attention Index
                                            </a>
                                        )}
                                        {queryDataUrl && (
                                            <a href={queryDataUrl} download="query_info.csv" className="download-csv-link" style={{ color: 'rgb(120, 180, 240)', textDecoration: 'none', display: 'inline-block', marginTop: '1rem' }}>
                                                Download Query Info
                                            </a>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    <div ref={msgEnd}></div>
                </div>
                <div className="chatFooter">
                        <div className="inp">
                            <textarea
                                className="growing-textarea"
                                placeholder="Send a message..."
                                value={inputString}
                                ref={inputRef}
                                onChange={(e) => { setInputString(e.target.value); }}
                                onKeyPress={handleKeyPress}
                                disabled={botIsTyping}
                            />
                            <button className='send' onClick={handleSend}>
                                <img src={sendBtn} alt="send"/>
                            </button>
                        </div>
                        <p>FenUI is more prone to mistakes if a label, start date, end date, and transform is not clearly specified.</p>
                    </div>
                    {showPopup && selectedData && (
                        <div className="popup">
                            <div className="popup-inner">
                                <h3 style={{ color: 'rgba(60, 120, 180)'}}>{selectedData.date}</h3>
                                <h2 style={{ marginTop: '1rem', marginBottom: '1rem'}}>{selectedData.headline}</h2>
                                <p style={{ fontSize: '1.2rem', lineHeight: '2.2rem'}}>{selectedData.document}</p>
                                <button onClick={() => setShowPopup(false)}>Close</button>
                            </div>
                        </div>
                    )}
            </div>
        </div>
    );
}

export default App;