package com.deep.framework.ast;

import com.deep.framework.ast.expression.*;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.ast.lexer.TokenType.*;
import static javax.lang.model.SourceVersion.isIdentifier;

public class Parser {
    String string = "\"";
    String strings = "\\\"";
    List<Node> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type.getToken());
        } else if (a.startsWith(strings)) {
            list.add(new StringExpression(a));
        } else {
            list.add(new Name(a));
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            if (a.equals(string)) {
                return a.concat(b);
            } else if (a.startsWith(string)) {
                if (a.concat(b).endsWith(strings)) return a.concat(b);
                if (!a.concat(b).endsWith(string)) return a.concat(b);
                add(a.concat(b));
                return "";
            } else if (Character.isWhitespace(b.charAt(0))) {
                add(a);
                return "";
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.contains(a.concat(b))) {
                return a.concat(b);
            }else if (a.concat(b).equals("..")) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });

        CompilationUnit compilationUnit = new CompilationUnit();
        parserBlockStatement(compilationUnit);
        parserStatement(compilationUnit);
        reduce(compilationUnit);
    }

    private void parserBlockStatement(Node prarent) {
        Node node = new Node(prarent);
        prarent.getChildrens().add(node);
        for (Node o : list) {
            if (o.equals(LBRACE)) {
                Node child = new BlockStatement(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(LPAREN)) {
                Node child = new ParametersExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RBRACK)) {
                node = node.getPrarent();
            } else {
                node.getChildrens().add(o);
            }
        }
    }

    public void parserStatement(Node prarent) {
        NodeList<Node> list = new NodeList<>();
        Node node = new Statement(prarent);
        for (Node o : prarent.getChildrens()) {
            if (o instanceof BlockStatement) {
                list.addAll(List.of(node, o));
                node = new Statement(prarent);
                parserStatement(o);
            } else if (o.equals(SEMI)) {
                list.add(node);
                node = new Statement(prarent);
            } else {
                parserStatement(o);
                node.getChildrens().add(o);
            }
        }
        if (list.isEmpty()) return;
        prarent.setChildrens(list);
    }

    public void reduce(Node node) {
        TypeParametersExpression.parser(node);
        Name.parser(node);
        ForStatement.parser(node);
        WhileStatement.parser(node);
        IfStatement.parser(node);
        /*ClassOrInterfaceDeclaration.parser(node);
        BinaryExpression.parser(node);
        AssignExpression.parser(node);*/
        for (Object n : node.getChildrens()) {
            if (n instanceof Node) {
                reduce((Node) n);
            }
        }
    }
}