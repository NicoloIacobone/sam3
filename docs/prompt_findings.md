Test effettuati:
- usare come prompt "all objects", "objects", "everything", ecc.. -> no objects detected.
    - SAM 3 è stato concepito per riconoscere "concetti visivi atomici" ed è vincolato all'uso di semplici frasi nominali (NPs), costituite da un sostantivo e da modificatori opzionali (ad esempio, "red apple" o "striped cat"). Non è progettato per query complesse, spaziali o che richiedono ragionamento logico.
    - gli autori sconsigliano l'uso di liste piatte di categorie; suggeriscono invece di definire il vocabolario attraverso un'ontologia ben curata e strutturata in gerarchie. La pratica più efficace per aumentare la precisione prevede l'inserimento esplicito di "hard negatives" (distrattori) nella lista. Introdurre categorie semanticamente vicine (ma assenti nell'immagine)


dataset LVIS (Large Vocabulary Instance Segmentation) 1200 categorie

- [X] Scaricare 1 scena di ScanNet
- [X] Analizzare la scena
- [ ] Capire che tipo di DataLoader utilizzare
- [ ] Capire come utilizzare LVIS (chiedi a Mattia)
- [ ] Ha senso usare VGGT-Omega visto che non è stato rilasciato il codice di training e gli autori dicono che non pianificano di rilasciarlo?

Obiettivo: creare un framework riproducibile, basato su una sola scena, closed vocabulary (passare a SAM una lista di categorie). Non pensare a risolvere problemi che non esistono, non c'è bisogno di pensare già alla generalizzazione su altre scene o altri dataset.