#[cfg(test)]
mod tests {
    use crate::anthropic::client::AnthropicClient;
    use crate::anthropic::models::{ChatRequest, ApiMessage};
    use crate::config::AnthropicConfig;
    use crate::utils::error::AgentError;
    use tokio::net::TcpListener;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use std::time::Duration;

    fn create_test_config() -> AnthropicConfig {
        AnthropicConfig {
            api_key: "test-key".to_string(),
            base_url: "http://localhost:8080".to_string(),
            model: "claude-test".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 5, // Short timeout for tests
            max_retries: 2, // Fewer retries for tests
        }
    }

    fn create_test_request() -> ChatRequest {
        ChatRequest {
            model: "claude-test".to_string(),
            max_tokens: 100,
            messages: vec![ApiMessage::user("Hello")],
            system: None,
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            stream: None,
        }
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = create_test_config();
        let client = AnthropicClient::new(config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_client_creation_with_invalid_timeout() {
        let mut config = create_test_config();
        config.timeout_seconds = 0; // Invalid timeout
        
        // This should still work as reqwest handles 0 timeout gracefully
        let client = AnthropicClient::new(config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_connection_timeout() {
        let mut config = create_test_config();
        config.base_url = "http://192.0.2.1:80".to_string(); // Non-routable IP for timeout
        config.timeout_seconds = 1; // Very short timeout
        
        let client = AnthropicClient::new(config).unwrap();
        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AgentError::Http(_) | AgentError::AnthropicApi { .. } => {
                // Expected - either HTTP error or API error
            }
            _ => panic!("Expected HTTP or API error for timeout"),
        }
    }

    #[tokio::test]
    async fn test_invalid_url_error() {
        let mut config = create_test_config();
        config.base_url = "not-a-valid-url".to_string();
        
        let client = AnthropicClient::new(config).unwrap();
        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_authentication_error_handling() {
        // Start a mock server that returns 401
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let response = "HTTP/1.1 401 Unauthorized\r\n\
                           Content-Type: application/json\r\n\
                           Content-Length: 60\r\n\
                           \r\n\
                           {\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\"}}";
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);
        
        let client = AnthropicClient::new(config).unwrap();
        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AgentError::Authentication { .. } => {
                // Expected authentication error
            }
            other => panic!("Expected authentication error, got: {:?}", other),
        }

        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_rate_limit_error_handling() {
        // Start a mock server that returns 429
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let response = "HTTP/1.1 429 Too Many Requests\r\n\
                           Content-Type: application/json\r\n\
                           Content-Length: 65\r\n\
                           \r\n\
                           {\"error\":{\"message\":\"Rate limit exceeded\",\"type\":\"rate_limit_error\"}}";
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);
        
        let client = AnthropicClient::new(config).unwrap();
        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AgentError::RateLimit { .. } => {
                // Expected rate limit error
            }
            other => panic!("Expected rate limit error, got: {:?}", other),
        }

        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_server_error_with_retry() {
        // Start a mock server that returns 500 then 200
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            // First request - return 500
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let error_response = "HTTP/1.1 500 Internal Server Error\r\n\
                                 Content-Type: application/json\r\n\
                                 Content-Length: 55\r\n\
                                 \r\n\
                                 {\"error\":{\"message\":\"Server error\",\"type\":\"server_error\"}}";
            socket.write_all(error_response.as_bytes()).await.unwrap();

            // Second request - return success
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let success_response = "HTTP/1.1 200 OK\r\n\
                                   Content-Type: application/json\r\n\
                                   Content-Length: 200\r\n\
                                   \r\n\
                                   {\"id\":\"test\",\"model\":\"claude-test\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hello!\"}],\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}";
            socket.write_all(success_response.as_bytes()).await.unwrap();
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);
        config.max_retries = 3; // Allow retries

        let client = AnthropicClient::new(config).unwrap();

        // Reset circuit breaker to ensure clean test state
        client.reset_circuit_breaker().await;

        let request = create_test_request();

        let result = client.chat(request).await;
        assert!(result.is_ok(), "Expected successful retry after server error");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_overloaded_error_handling() {
        // Start a mock server that returns 529 (overloaded)
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let response = "HTTP/1.1 529 Service Overloaded\r\n\
                           Content-Type: application/json\r\n\
                           Content-Length: 70\r\n\
                           \r\n\
                           {\"error\":{\"message\":\"Service temporarily overloaded\",\"type\":\"overloaded_error\"}}";
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);
        config.max_retries = 1; // Only one retry to speed up test

        let client = AnthropicClient::new(config).unwrap();

        // Reset circuit breaker to ensure clean test state
        client.reset_circuit_breaker().await;

        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AgentError::AnthropicApi { message } => {
                assert!(message.contains("overloaded") || message.contains("529"));
            }
            other => panic!("Expected API overloaded error, got: {:?}", other),
        }

        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_malformed_json_response() {
        // Start a mock server that returns invalid JSON
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = socket.read(&mut buffer).await;

            let response = "HTTP/1.1 200 OK\r\n\
                           Content-Type: application/json\r\n\
                           Content-Length: 20\r\n\
                           \r\n\
                           {invalid json here}";
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);

        let client = AnthropicClient::new(config).unwrap();

        // Reset circuit breaker to ensure clean test state
        client.reset_circuit_breaker().await;

        let request = create_test_request();
        
        let result = client.chat(request).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AgentError::Json(_) => {
                // Expected JSON parsing error
            }
            other => panic!("Expected JSON error, got: {:?}", other),
        }

        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_config_update() {
        let config = create_test_config();
        let mut client = AnthropicClient::new(config).unwrap();
        
        let mut new_config = create_test_config();
        new_config.timeout_seconds = 10;
        new_config.model = "claude-new".to_string();
        
        let result = client.update_config(new_config.clone());
        assert!(result.is_ok());
        assert_eq!(client.config().timeout_seconds, 10);
        assert_eq!(client.config().model, "claude-new");
    }

    #[tokio::test]
    async fn test_retry_mechanism_exhaustion() {
        // Start a mock server that always returns 500
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            // Accept multiple connections and always return 500
            for _ in 0..3 {
                if let Ok((mut socket, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = socket.read(&mut buffer).await;

                    let error_response = "HTTP/1.1 500 Internal Server Error\r\n\
                                         Content-Type: application/json\r\n\
                                         Content-Length: 55\r\n\
                                         \r\n\
                                         {\"error\":{\"message\":\"Server error\",\"type\":\"server_error\"}}";
                    let _ = socket.write_all(error_response.as_bytes()).await;
                }
            }
        });

        let mut config = create_test_config();
        config.base_url = format!("http://{}", addr);
        config.max_retries = 1; // Minimal retries for faster test
        config.timeout_seconds = 5; // Shorter timeout

        let client = AnthropicClient::new(config).unwrap();

        // Reset circuit breaker to ensure clean test state
        client.reset_circuit_breaker().await;

        let request = create_test_request();

        let start_time = std::time::Instant::now();
        let result = client.chat(request).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_err());
        // Should have taken some time due to retries with backoff, but not too long
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed <= Duration::from_secs(10)); // Reasonable upper bound

        server.await.unwrap();
    }
}
