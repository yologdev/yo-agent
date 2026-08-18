#![cfg(feature = "openapi")]

use yoagent::openapi::{OpenApiConfig, OpenApiToolAdapter, OperationFilter};
use yoagent::types::{AgentTool, ToolContext};

const SPEC: &str = r#"{
    "openapi": "3.0.0",
    "info": { "title": "Test API", "version": "1.0.0" },
    "servers": [{ "url": "https://api.example.com/v1" }],
    "paths": {
        "/items": {
            "get": {
                "operationId": "listItems",
                "summary": "List all items",
                "description": "Returns a paginated list of items.",
                "tags": ["items"],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "required": false,
                        "schema": { "type": "integer" }
                    },
                    {
                        "name": "offset",
                        "in": "query",
                        "required": false,
                        "schema": { "type": "integer" }
                    }
                ],
                "responses": { "200": { "description": "A list of items" } }
            },
            "post": {
                "operationId": "createItem",
                "summary": "Create an item",
                "tags": ["items"],
                "requestBody": {
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "price": { "type": "number" }
                                },
                                "required": ["name"]
                            }
                        }
                    }
                },
                "responses": { "201": { "description": "Item created" } }
            }
        },
        "/items/{itemId}": {
            "get": {
                "operationId": "getItem",
                "summary": "Get an item",
                "tags": ["items"],
                "parameters": [
                    {
                        "name": "itemId",
                        "in": "path",
                        "required": true,
                        "schema": { "type": "string" }
                    }
                ],
                "responses": { "200": { "description": "An item" } }
            }
        },
        "/users": {
            "get": {
                "operationId": "listUsers",
                "summary": "List users",
                "tags": ["users"],
                "parameters": [
                    {
                        "name": "X-Request-Id",
                        "in": "header",
                        "required": false,
                        "schema": { "type": "string" }
                    }
                ],
                "responses": { "200": { "description": "A list of users" } }
            }
        }
    }
}"#;

fn test_ctx() -> ToolContext {
    ToolContext::new("tc-1", "test")
}

// ---------------------------------------------------------------------------
// Spec parsing / schema tests (no HTTP)
// ---------------------------------------------------------------------------

#[test]
fn test_adapter_creation() {
    let adapters =
        OpenApiToolAdapter::from_str(SPEC, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    assert_eq!(adapters.len(), 4);
}

#[test]
fn test_tool_names() {
    let adapters =
        OpenApiToolAdapter::from_str(SPEC, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let names: Vec<&str> = adapters.iter().map(|a| a.name()).collect();
    assert!(names.contains(&"listItems"));
    assert!(names.contains(&"createItem"));
    assert!(names.contains(&"getItem"));
    assert!(names.contains(&"listUsers"));
}

#[test]
fn test_tool_description() {
    let adapters =
        OpenApiToolAdapter::from_str(SPEC, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();
    assert_eq!(list_items.label(), "List all items");
    assert_eq!(
        list_items.description(),
        "Returns a paginated list of items."
    );
}

#[test]
fn test_schema_output() {
    let adapters =
        OpenApiToolAdapter::from_str(SPEC, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let get_item = adapters.iter().find(|a| a.name() == "getItem").unwrap();
    let schema = get_item.parameters_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["itemId"].is_object());
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("itemId")));
}

#[test]
fn test_filter_by_tag() {
    let filter = OperationFilter::ByTag(vec!["users".into()]);
    let adapters = OpenApiToolAdapter::from_str(SPEC, OpenApiConfig::default(), &filter).unwrap();
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0].name(), "listUsers");
}

#[test]
fn test_name_prefix() {
    let config = OpenApiConfig::default().with_name_prefix("myapi");
    let adapters = OpenApiToolAdapter::from_str(SPEC, config, &OperationFilter::All).unwrap();
    assert!(adapters.iter().all(|a| a.name().starts_with("myapi__")));
}

// ---------------------------------------------------------------------------
// execute() tests with wiremock
// ---------------------------------------------------------------------------

fn make_spec(base_url: &str) -> String {
    format!(
        r#"{{
        "openapi": "3.0.0",
        "info": {{ "title": "Test", "version": "1.0.0" }},
        "servers": [{{ "url": "{base_url}" }}],
        "paths": {{
            "/items": {{
                "get": {{
                    "operationId": "listItems",
                    "parameters": [
                        {{ "name": "limit", "in": "query", "schema": {{ "type": "integer" }} }},
                        {{ "name": "X-Trace", "in": "header", "schema": {{ "type": "string" }} }}
                    ],
                    "responses": {{ "200": {{ "description": "ok" }} }}
                }},
                "post": {{
                    "operationId": "createItem",
                    "requestBody": {{
                        "required": true,
                        "content": {{
                            "application/json": {{
                                "schema": {{ "type": "object", "properties": {{ "name": {{ "type": "string" }} }} }}
                            }}
                        }}
                    }},
                    "responses": {{ "201": {{ "description": "created" }} }}
                }}
            }},
            "/items/{{itemId}}": {{
                "get": {{
                    "operationId": "getItem",
                    "parameters": [
                        {{ "name": "itemId", "in": "path", "required": true, "schema": {{ "type": "string" }} }}
                    ],
                    "responses": {{ "200": {{ "description": "ok" }} }}
                }}
            }}
        }}
    }}"#
    )
}

#[tokio::test]
async fn test_execute_get_with_path_param() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items/42"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"id":"42"}"#))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let get_item = adapters.iter().find(|a| a.name() == "getItem").unwrap();

    let result = get_item
        .execute(serde_json::json!({"itemId": "42"}), test_ctx())
        .await
        .unwrap();

    let text = match &result.content[0] {
        yoagent::types::Content::Text { text } => text,
        _ => panic!("Expected text content"),
    };
    assert!(text.contains("200"), "Should contain status 200");
    assert!(
        text.contains(r#"{"id":"42"}"#),
        "Should contain response body"
    );
    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_get_with_query_params() {
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items"))
        .and(query_param("limit", "10"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({"limit": 10}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_post_with_body() {
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/items"))
        .and(body_json(serde_json::json!({"name": "Widget"})))
        .respond_with(ResponseTemplate::new(201).set_body_string(r#"{"id":"1","name":"Widget"}"#))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let create_item = adapters.iter().find(|a| a.name() == "createItem").unwrap();

    let result = create_item
        .execute(serde_json::json!({"body": {"name": "Widget"}}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 201);
}

#[tokio::test]
async fn test_execute_with_bearer_auth() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items"))
        .and(header("Authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let config = OpenApiConfig::default().with_bearer_token("test-token");
    let adapters = OpenApiToolAdapter::from_str(&spec, config, &OperationFilter::All).unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_with_api_key_auth() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items"))
        .and(header("X-API-Key", "my-key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let config = OpenApiConfig::default().with_api_key("X-API-Key", "my-key");
    let adapters = OpenApiToolAdapter::from_str(&spec, config, &OperationFilter::All).unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_with_custom_headers() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items"))
        .and(header("X-Custom", "custom-value"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let config = OpenApiConfig::default().with_header("X-Custom", "custom-value");
    let adapters = OpenApiToolAdapter::from_str(&spec, config, &OperationFilter::All).unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_with_header_param() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items"))
        .and(header("X-Trace", "trace-123"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({"X-Trace": "trace-123"}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}

#[tokio::test]
async fn test_execute_non_2xx_returns_ok() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/items/999"))
        .respond_with(ResponseTemplate::new(404).set_body_string("Not found"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let get_item = adapters.iter().find(|a| a.name() == "getItem").unwrap();

    // Non-2xx should still return Ok so the LLM can reason about the error
    let result = get_item
        .execute(serde_json::json!({"itemId": "999"}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 404);
    let text = match &result.content[0] {
        yoagent::types::Content::Text { text } => text,
        _ => panic!("Expected text"),
    };
    assert!(text.contains("Not found"));
}

#[tokio::test]
async fn test_execute_response_truncation() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let long_body = "x".repeat(1000);
    Mock::given(method("GET"))
        .and(path("/items"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&long_body))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let config = OpenApiConfig::default().with_max_response_bytes(100);
    let adapters = OpenApiToolAdapter::from_str(&spec, config, &OperationFilter::All).unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    let result = list_items
        .execute(serde_json::json!({}), test_ctx())
        .await
        .unwrap();

    let text = match &result.content[0] {
        yoagent::types::Content::Text { text } => text,
        _ => panic!("Expected text"),
    };
    assert!(text.contains("[truncated]"));
    // The body portion should be truncated, not the full 1000 chars
    assert!(text.len() < 500);
}

#[tokio::test]
async fn test_execute_missing_path_param_errors() {
    use wiremock::MockServer;

    let server = MockServer::start().await;
    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let get_item = adapters.iter().find(|a| a.name() == "getItem").unwrap();

    // Missing required path param returns error as content so LLM can self-correct
    let result = get_item
        .execute(serde_json::json!({}), test_ctx())
        .await
        .unwrap();
    let text = match &result.content[0] {
        yoagent::types::Content::Text { text } => text,
        _ => panic!("Expected text"),
    };
    assert!(text.contains("Missing required path parameter"));
    assert!(text.contains("itemId"));
}

#[tokio::test]
async fn test_execute_rejects_non_object_params() {
    use wiremock::MockServer;

    let server = MockServer::start().await;
    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let list_items = adapters.iter().find(|a| a.name() == "listItems").unwrap();

    // Non-object params return error as content so LLM can self-correct
    let result = list_items
        .execute(serde_json::json!("not an object"), test_ctx())
        .await
        .unwrap();
    let text = match &result.content[0] {
        yoagent::types::Content::Text { text } => text,
        _ => panic!("Expected text"),
    };
    assert!(text.contains("string"));
}

#[tokio::test]
async fn test_execute_path_param_url_encoded() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // The path param value "a/b" should be URL-encoded to "a%2Fb"
    Mock::given(method("GET"))
        .and(path("/items/a%2Fb"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let spec = make_spec(&server.uri());
    let adapters =
        OpenApiToolAdapter::from_str(&spec, OpenApiConfig::default(), &OperationFilter::All)
            .unwrap();
    let get_item = adapters.iter().find(|a| a.name() == "getItem").unwrap();

    let result = get_item
        .execute(serde_json::json!({"itemId": "a/b"}), test_ctx())
        .await
        .unwrap();

    assert_eq!(result.details["status"], 200);
}
