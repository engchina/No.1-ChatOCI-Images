SELECT 
    ie.ID as embed_id, 
    ie.BUCKET as bucket,
    ie.OBJECT_NAME as object_name,
    VECTOR_DISTANCE(ie.EMBEDDING, (
        SELECT  
            TO_VECTOR(et.embed_vector) embed_vector 
        FROM 
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS( 
                'hello world', 
                JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}')) t, 
                JSON_TABLE(t.column_value, '$[*]' 
                COLUMNS( 
                    embed_id NUMBER PATH '$.embed_id', 
                    embed_data VARCHAR2(4000) PATH '$.embed_data', 
                    embed_vector CLOB PATH '$.embed_vector' 
                ) 
            ) 
        et), COSINE 
    ) vector_distance 
FROM  
    IMG_EMBEDDINGS ie 
WHERE  
    1 = 1 
    AND VECTOR_DISTANCE(ie.EMBEDDING, ( 
        SELECT  
            TO_VECTOR(et.embed_vector) embed_vector 
        FROM 
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS( 
                'hello world', 
                JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}')) t, 
                JSON_TABLE(t.column_value, '$[*]' 
                COLUMNS( 
                    embed_id NUMBER PATH '$.embed_id', 
                    embed_data VARCHAR2(4000) PATH '$.embed_data', 
                    embed_vector CLOB PATH '$.embed_vector' 
                ) 
            ) 
        et), COSINE 
    ) <= 0.7 
ORDER BY  
    vector_distance
FETCH FIRST 10 ROWS ONLY;

SELECT 
    ie.ID as embed_id, 
    ie.BUCKET as bucket,
    ie.OBJECT_NAME as object_name,
    VECTOR_DISTANCE(ie.EMBEDDING, (
        SELECT  
            TO_VECTOR(et.embed_vector) embed_vector 
        FROM 
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS( 
                'hello world', 
                JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}')) t, 
                JSON_TABLE(t.column_value, '$[*]' 
                COLUMNS( 
                    embed_id NUMBER PATH '$.embed_id', 
                    embed_data VARCHAR2(4000) PATH '$.embed_data', 
                    embed_vector CLOB PATH '$.embed_vector' 
                ) 
            ) 
        et), COSINE 
    ) vector_distance 
FROM  
    IMG_EMBEDDINGS ie 
ORDER BY  
    vector_distance
FETCH FIRST 10 ROWS ONLY;