db.createCollection("textos")

db.textos.insertOne({fragment:"cercar textes en MongoDB por ser complicat"})
db.textos.insertOne({fragment:"cercar textes en MongoDB por ser divertit"})
db.textos.insertOne({fragment:"cercar textes en MongoDB podria ser més senzill"})

db.textos.createIndex({fragment:"text"})

db.textos.find({$text:{$search:"MongoDB"}},{punts:{$meta:"textScore"}})

//Varies paraules separades per espai -> for (let i = 0; i < 
db.textos.find({$text:{$search:"divertit podria"}})

//Frase complerta:
db.textos.find({$text:{$search:"\"MongoDB podria\""}})

//Que no aparegui alguna paraula
db.textos.find({$text:{$search:"MongoDB -podria"}})

//Case sensitive
db.textos.find({$text:{$search:"mongoDB -podria", $caseSensitive: true}})
