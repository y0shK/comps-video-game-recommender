import React, {useEffect, useState} from 'react';

import './App.css';

// https://stackoverflow.com/questions/72022246/how-to-pass-a-flask-post-to-react
// https://stackoverflow.com/questions/55757761/handle-an-input-with-react-hooks

function App() {

    const [query, setQuery] = useState('')
    // const [inputValue, setInputValue] = React.useState("");

    const onChangeHandler = event => {
        setQuery(event.target.value);
     };

    const [backendData, setBackendData] = useState([{}]);

    function handleTransfer() {
        fetch('/query', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({"query_body": query})
        }).then(() => {
            console.log("POST: " + query)
        })
  
        fetch('/query').then(
          response => response.json()
        ).then(data => setBackendData(data)).then(() => {
            console.log("GET: " + data)
        })
    }

    return (
        
        <div className="container">
        <h1>Query</h1>
        <h3>{query}</h3>
       
        <input
   type="text"
   name="name"
   onChange={onChangeHandler}
   value={query}
/>
        
        <button onClick={handleTransfer}>Transfer Data</button>
        <h3>{backendData.query_body}</h3>
        </div>

    );
}

export default App;
