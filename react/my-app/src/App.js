import React, {useEffect, useState, useCallback} from 'react';

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

    const [recName, setRecName] = useState("");
    const [recImg, setRecImg] = useState("");

    // useEffect(() => { https://stackoverflow.com/questions/56247433/how-to-use-setstate-callback-on-react-hooks

    // https://stackoverflow.com/questions/72113386/reactjs-how-do-you-call-an-async-fuction-on-button-click
    // https://stackoverflow.com/questions/72113386/reactjs-how-do-you-call-an-async-fuction-on-button-click
    // https://stackoverflow.com/questions/71174509/proper-async-await-syntax-for-fetching-data-using-useeffect-hook-in-react
    // https://stackoverflow.com/questions/56663785/invalid-hook-call-hooks-can-only-be-called-inside-of-the-body-of-a-function-com

     // https://stackoverflow.com/questions/40825010/const-or-let-in-react-component
    let handleTransfer = useCallback( async () => {

    // async function handleTransferHelper() {
        fetch('/query', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({"query_body": query})
        }).then(() => {
            console.log("POST: " + query)
        })
          
        // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
        // https://stackoverflow.com/questions/54936559/using-async-await-in-react-component
        const response = await fetch("/query"); 
        const games = await response.json();
            console.log(games);
             setRecName(games.game)
             setRecImg(games.url)

            
            console.log(recName)
            console.log(recImg)

            console.log(games['game'])
            console.log(games.game)
            console.log(recName)
            console.log(recImg)
    })
    // handleTransferHelper()
    // }, []);

    // https://stackoverflow.com/questions/51184136/display-an-image-from-url-in-reactjs

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
        <h3>{recName}</h3>
        <img 
      src={recImg}
      alt="boxart"
      />
        </div>

    );


}

            //useEffect(() => { setRecName(games.game) }, [])
            //useEffect(() => { setRecImg(games.url) }, [])
  
        // fetch('/query').then(
          // response => response.json()
        // ).then(data => setBackendData(data)).then((data) => {
           // console.log("GET: " + data)
        // })

    

export default App;
